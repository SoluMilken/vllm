# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AOT (Ahead-Of-Time) compilation support for vLLM.

This module owns everything specific to the AOT compile path:

- :class:`AotCompileCache` – on-disk cache (hash, load, save).
- :class:`AotRunner`       – runtime state machine (hot path, disk-cache hit,
  compile-from-scratch, reset for elastic EP).
- Private helpers used only by the two classes above.
"""

import hashlib
import inspect
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.compilation.compiler_interface import get_inductor_factors
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.utils import hash_factors
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

if TYPE_CHECKING:
    try:
        from torch._dynamo.package import SourceInfo
    except ImportError:
        SourceInfo = Any  # type: ignore[assignment, misc]

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_code_hash_with_content(file_contents: dict[str, str]) -> str:
    items = list(sorted(file_contents.items(), key=lambda x: x[0]))
    hash_content = []
    for filepath, content in items:
        hash_content.append(filepath)
        if filepath == "<string>":
            # This means the function was dynamically generated, with
            # e.g. exec(). We can't actually check these.
            continue
        hash_content.append(content)
    result: str = safe_hash(
        "\n".join(hash_content).encode(), usedforsecurity=False
    ).hexdigest()
    return result


def _compute_code_hash(files: set[str]) -> str:
    logger.debug(
        "Traced files (to be considered for compilation cache):\n%s",
        "\n".join(files),
    )
    file_contents = {}
    for filepath in files:
        # Skip files that don't exist (e.g., <string>, <frozen modules>, etc.)
        if not os.path.isfile(filepath):
            file_contents[filepath] = ""
        else:
            with open(filepath) as f:
                file_contents[filepath] = f.read()
    return _compute_code_hash_with_content(file_contents)


# ---------------------------------------------------------------------------
# AotCompileCache
# ---------------------------------------------------------------------------


class AotCompileCache:
    """Manages the on-disk cache for AOT (Ahead-Of-Time) compiled models.

    Background
    ----------
    vLLM can use ``torch.compile`` in AOT mode to compile a model once and
    serialize the result to disk.  On the next startup the compiled artifact
    is loaded directly, skipping the expensive compilation step entirely.

    This class owns three responsibilities:

    1. **Path computation** (``__init__``): deterministically derive the cache
       directory from the vLLM config + the model's ``forward`` function so
       that a different model or config automatically maps to a different path.

    2. **Loading** (``try_load``): deserialize a previously saved artifact,
       verify that no source files have changed since it was compiled, and
       return the ready-to-call compiled function.

    3. **Saving** (``save``): atomically write a freshly compiled function to
       the cache directory so future runs can skip compilation.

    Typical call flow
    -----------------
    ::

        cache = AotCompileCache(vllm_config, model.forward)

        compiled_fn = cache.try_load(model)  # fast path: load from disk
        if compiled_fn is None:
            compiled_fn = compile(model)  # slow path: actually compile
            cache.save(compiled_fn)
    """

    def __init__(self, vllm_config: VllmConfig, forward_fn: Callable[..., Any]) -> None:
        """Compute and store the cache path for this (config, model) pair.

        The directory is derived from a hash of:
        - environment variables that affect compilation (e.g. PP layer partition)
        - the full ``VllmConfig`` (model size, parallelism, quantization, …)
        - the identity of ``forward_fn`` (version + qualified name + line number)

        Each tensor-parallel rank gets its own sub-directory because compiled
        artifacts can differ across ranks (e.g. different weight shards).
        """
        hash_key = AotCompileCache._compute_hash_key(vllm_config, forward_fn)
        rank = vllm_config.parallel_config.rank
        dp_rank = vllm_config.parallel_config.data_parallel_index
        self.cache_dir = os.path.join(
            envs.VLLM_CACHE_ROOT,
            "torch_compile_cache",
            "torch_aot_compile",
            hash_key,
            f"rank_{rank}_{dp_rank}",
        )
        self.artifact_path = os.path.join(self.cache_dir, "model")
        self._vllm_config = vllm_config
        self._loaded_from_disk = False

    def try_load(self, model: torch.nn.Module) -> Any:
        """Try to load an AOT-compiled function from disk.

        Steps performed on a cache hit:

        1. Deserialize the compiled artifact using PyTorch's
           ``load_compiled_function``.  ``model.forward.__globals__`` is
           passed so that global symbols (custom ops, helpers, …) that were
           captured during the original trace can be resolved correctly.
        2. Verify that none of the traced source files have changed since the
           artifact was created (``_verify_source_unchanged``).
        3. Optionally disable shape guards if the config says so.
        4. Finalize the compiled backend (loads Inductor kernels, etc.).

        Returns the loaded callable on success, or ``None`` on any failure so
        the caller can fall back to recompilation.  Re-raises if
        ``VLLM_FORCE_AOT_LOAD`` is set, which is useful for CI environments
        that must never recompile.
        """
        from vllm.compilation.decorators import maybe_use_cudagraph_partition_wrapper
        from vllm.compilation.monitor import monitor_torch_compile

        try:
            with monitor_torch_compile(self._vllm_config):
                with (
                    set_current_vllm_config(self._vllm_config),
                    open(self.artifact_path, "rb") as f,
                ):
                    loaded_fn = torch.compiler.load_compiled_function(
                        f, f_globals=model.forward.__globals__
                    )
                AotCompileCache._verify_source_unchanged(
                    loaded_fn.source_info(), self._vllm_config
                )
                ds_config = self._vllm_config.compilation_config.dynamic_shapes_config
                if not ds_config.evaluate_guards:
                    loaded_fn.disable_guard_check()
                with maybe_use_cudagraph_partition_wrapper(self._vllm_config):
                    loaded_fn._artifacts.compiled_fn.finalize_loading(self._vllm_config)
            compilation_counter.num_aot_artifacts_loaded += 1
            self._loaded_from_disk = True
            logger.info(
                "Directly load AOT compilation from path %s", self.artifact_path
            )
            return loaded_fn
        except Exception as e:
            if os.path.exists(self.artifact_path):
                message = (
                    "Compile cache file corrupted."
                    if isinstance(e, EOFError)
                    else str(e)
                )
                logger.warning(
                    "Compiling model again due to a load failure from %s, reason: %s",
                    self.artifact_path,
                    message,
                )
            if envs.VLLM_FORCE_AOT_LOAD:
                raise e
            return None

    def save(self, compiled_fn: Any) -> None:
        """Persist a freshly compiled function to disk.

        Skipped entirely if:
        - the compile cache is disabled (``VLLM_DISABLE_COMPILE_CACHE``)
        - the artifact was itself loaded from disk (no point re-saving it)

        The write is atomic: the artifact is first serialized to a temporary
        file ``<path>.<pid>.tmp`` and then renamed into place, so a crash
        during saving never leaves a corrupt cache entry.
        """
        if envs.VLLM_DISABLE_COMPILE_CACHE:
            return
        if self._loaded_from_disk:
            logger.debug("AOT compiled function was loaded from cache, skipping save")
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # File saving should be atomic, so we will save to a temporary location
            # first. Should be upstreamed to PyTorch 2.12 as well.
            tmp_file = f"{self.artifact_path}.{os.getpid()}.tmp"
            compiled_fn.save_compiled_function(tmp_file)
            os.replace(tmp_file, self.artifact_path)
            compilation_counter.num_aot_artifacts_saved += 1
            logger.info_once(
                "saved AOT compiled function to %s",
                self.artifact_path,
                scope="local",
            )
        except Exception as e:
            logger.warning(
                "unable to save AOT compiled function to %s: %s",
                self.artifact_path,
                e,
            )

    @staticmethod
    def _compute_hash_key(
        vllm_config: VllmConfig, forward_fn: Callable[..., Any]
    ) -> str:
        """Derive a deterministic cache key from config + model identity.

        Combines all factors that can change the compiled output:

        - environment variables (e.g. ``VLLM_PP_LAYER_PARTITION``)
        - the full ``VllmConfig`` (model size, parallelism, quantization, …)
        - Inductor compile factors (if ``VLLM_USE_MEGA_AOT_ARTIFACT``)
        - the identity of the ``forward`` function (version + name + line number)
        """
        factors: list[str] = [
            hash_factors(envs.compile_factors()),
            vllm_config.compute_hash(),
        ]
        if envs.VLLM_USE_MEGA_AOT_ARTIFACT:
            factors.extend(get_inductor_factors())
        factors.append(AotCompileCache._model_hash_key(forward_fn))
        return hashlib.sha256(str(factors).encode()).hexdigest()

    @staticmethod
    def _model_hash_key(fn: Callable[..., Any]) -> str:
        """Produce a stable hash that uniquely identifies a ``forward`` method.

        The hash combines the vLLM version, the fully-qualified function name,
        and its line number in the source file.  This means that if the function
        is renamed, moved, or the library is upgraded, the hash changes and the
        old cache entry is ignored.
        """
        import vllm

        sha256_hash = hashlib.sha256()
        sha256_hash.update(vllm.__version__.encode())
        sha256_hash.update(fn.__qualname__.encode())
        sha256_hash.update(str(fn.__code__.co_firstlineno).encode())
        return sha256_hash.hexdigest()

    @staticmethod
    def _verify_source_unchanged(
        source_info: "SourceInfo", vllm_config: VllmConfig
    ) -> None:
        """Guard against loading a stale cache artifact.

        When a model is AOT-compiled, Dynamo records the content of every
        Python source file it traced through (``source_info.inlined_sources``).
        This method re-reads those files from disk and compares their checksums
        against what was recorded at compile time.

        Raises ``RuntimeError`` if any file has changed, which causes the
        caller to fall back to recompilation.
        """
        file_contents = {}
        for source in source_info.inlined_sources:
            module = sys.modules[source.module]
            file = inspect.getfile(module)
            vllm_config.compilation_config.traced_files.add(file)
            file_contents[file] = source.content
        expected_checksum = _compute_code_hash_with_content(file_contents)
        actual_checksum = _compute_code_hash(set(file_contents.keys()))
        if expected_checksum != actual_checksum:
            raise RuntimeError(
                "Source code has changed since the last compilation. "
                "Recompiling the model."
            )


# ---------------------------------------------------------------------------
# AotRunner
# ---------------------------------------------------------------------------


class AotRunner:
    """Manages the full AOT compile lifecycle for a single model instance.

    Owns all AOT-specific runtime state so the model's ``__call__`` does not
    need to hold it directly.

    Typical call flow (first invocation)
    -------------------------------------
    ::

        runner = AotRunner(vllm_config, model.forward)

        # hot path A: compiled_fn already in memory
        ok, out = runner.try_run_hot(model, *args, **kwargs)
        if ok:
            return out

        # hot path B: disk cache hit
        ok, out = runner.try_load_and_run(model, *args, **kwargs)
        if ok:
            return out

        # slow path: compile from scratch (inside dynamo context)
        ok, out = runner.compile_and_run(model, *args, **kwargs)

    Elastic EP reset
    ----------------
    When expert-parallelism is reconfigured, call ``runner.reset()`` to clear
    the stale compiled function and cache.  The cache will be recreated lazily
    on the next ``try_load_and_run`` / ``compile_and_run`` call using the
    updated ``vllm_config``.
    """

    def __init__(self, vllm_config: VllmConfig, forward_fn: Callable[..., Any]) -> None:
        self._vllm_config = vllm_config
        self._forward_fn = forward_fn
        self.compiled_fn: Any = None
        self._cache: AotCompileCache | None = None

    # ------------------------------------------------------------------
    # Cache accessor (lazy so reset() can invalidate it)
    # ------------------------------------------------------------------

    def _get_cache(self) -> AotCompileCache:
        if self._cache is None:
            self._cache = AotCompileCache(self._vllm_config, self._forward_fn)
        return self._cache

    # ------------------------------------------------------------------
    # Hot path A: compiled fn already in memory
    # ------------------------------------------------------------------

    def try_run_hot(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> tuple[bool, Any]:
        """If compiled_fn is already in memory, run it and return (True, output).

        The cudagraph partition wrapper must be active at runtime so CUDA graph
        capture works correctly with inductor graph partitioning.
        """
        from vllm.compilation.decorators import maybe_use_cudagraph_partition_wrapper

        if self.compiled_fn is None:
            return False, None
        with maybe_use_cudagraph_partition_wrapper(self._vllm_config):
            return True, self.compiled_fn(model, *args, **kwargs)

    # ------------------------------------------------------------------
    # Hot path B: first call, try disk cache
    # ------------------------------------------------------------------

    def try_load_and_run(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> tuple[bool, Any]:
        """Try to load from disk cache and run (first call only).

        Returns ``(True, output)`` on a cache hit, ``(False, None)`` otherwise.
        """
        from vllm.compilation.decorators import maybe_use_cudagraph_partition_wrapper
        from vllm.compilation.monitor import monitor_profiling_run

        if envs.VLLM_DISABLE_COMPILE_CACHE:
            return False, None

        loaded_fn = self._get_cache().try_load(model)
        if loaded_fn is None:
            return False, None

        self.compiled_fn = loaded_fn
        with (
            monitor_profiling_run(),
            maybe_use_cudagraph_partition_wrapper(self._vllm_config),
        ):
            return True, self.compiled_fn(model, *args, **kwargs)

    # ------------------------------------------------------------------
    # Slow path: compile from scratch
    # ------------------------------------------------------------------

    def compile_and_run(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> tuple[bool, Any]:
        """AOT-compile the model, save the artifact, and run it once.

        Must be called inside the dynamo/fx/inductor config patch context
        (i.e. inside the ``with`` block in ``__call__``).

        Returns ``(True, output)`` on success, or ``(False, None)`` when AOT
        compile should be skipped (e.g. eager backend) so the caller falls
        through to the normal ``torch.compile`` path.
        """
        from vllm.compilation.monitor import (
            monitor_profiling_run,
            monitor_torch_compile,
        )

        if self._vllm_config.compilation_config.backend == "eager":
            logger.warning("Detected eager backend, disabling AOT compile.")
            return False, None

        with monitor_torch_compile(self._vllm_config):
            self.compiled_fn = model.aot_compile(*args, **kwargs)
            compilation_counter.num_aot_compiles += 1
            self._get_cache().save(self.compiled_fn)

        with monitor_profiling_run():
            output = self.compiled_fn(model, *args, **kwargs)
        return True, output

    # ------------------------------------------------------------------
    # Elastic EP reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear AOT state after elastic EP reconfiguration.

        The compiled function is invalidated because Inductor kernels may have
        old parameters (e.g. expert_map size) baked in as compile-time
        constants.  The cache is also cleared so it will be re-created with the
        new ``vllm_config`` on the next call.
        """
        self.compiled_fn = None
        self._cache = None
