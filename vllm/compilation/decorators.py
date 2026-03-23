# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import inspect
from collections.abc import Callable, Generator
from typing import Any, TypeVar, overload
from unittest.mock import patch

import torch
import torch.nn as nn
from torch._dynamo.symbolic_convert import InliningInstructionTranslator

import vllm.envs as envs
from vllm.compilation.aot import AotRunner
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import (
    CompilationMode,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.config.compilation import DynamicShapesType
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import is_torch_equal_or_newer

from .monitor import monitor_torch_compile

logger = init_logger(__name__)

IGNORE_COMPILE_KEY = "_ignore_compile_vllm"

_T = TypeVar("_T", bound=nn.Module)


def should_torch_compile_mm_encoder(vllm_config: VllmConfig) -> bool:
    """Callable to be passed to `@support_torch_compile`'s `enable_if` argument."""
    return vllm_config.compilation_config.compile_mm_encoder


def ignore_torch_compile(cls: type[_T]) -> type[_T]:
    """
    A decorator to ignore support_torch_compile decorator
    on the class. This is useful when a parent class has
    a support_torch_compile decorator, but we don't want to
    compile the class `cls` that inherits the parent class.
    This only ignores compiling the forward of the class the
    decorator is applied to.

    If the parent has ignore_torch_compile but the child has
    support_torch_compile, the child will still be compiled.

    If the class has one or more submodules
    that have support_torch_compile decorator applied, compile will
    not be ignored for those submodules.
    """
    setattr(cls, IGNORE_COMPILE_KEY, True)
    return cls


def _should_ignore_torch_compile(cls: type[_T]) -> bool:
    """
    Check if the class should be ignored for torch.compile.
    """
    return getattr(cls, IGNORE_COMPILE_KEY, False)


@overload
def support_torch_compile(
    *,
    enable_if: Callable[[VllmConfig], bool] | None = None,
) -> Callable[[type[_T]], type[_T]]: ...


@overload
def support_torch_compile(
    *,
    dynamic_arg_dims: dict[str, int | list[int]] | None,
) -> Callable[[type[_T]], type[_T]]: ...


@overload
def support_torch_compile(
    *,
    mark_unbacked_dims: dict[str, int | list[int]] | None,
) -> Callable[[type[_T]], type[_T]]: ...


@overload
def support_torch_compile(
    *,
    dynamic_arg_dims: dict[str, int | list[int]] | None,
    mark_unbacked_dims: dict[str, int | list[int]] | None,
) -> Callable[[type[_T]], type[_T]]: ...


@overload
def support_torch_compile(cls: type[_T]) -> type[_T]: ...


def support_torch_compile(
    cls: type[_T] | None = None,
    *,
    dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    mark_unbacked_dims: dict[str, int | list[int]] | None = None,
    enable_if: Callable[[VllmConfig], bool] | None = None,
    is_encoder: bool = False,
    shape_invariants: Callable[..., None] = lambda *args, **kwargs: None,
) -> Callable[[type[_T]], type[_T]] | type[_T]:
    """
    A decorator to add support for compiling the forward method of a class.

    Usage 1: use directly as a decorator without arguments:

    ```python
    @support_torch_compile
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]): ...
    ```

    Usage 2: use as a decorator with arguments:

    ```python
    @support_torch_compile(dynamic_arg_dims={"x": 0, "y": 0})
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]): ...
    ```

    `dynamic_arg_dims` is a dictionary that maps argument names to the dynamic
    dimensions of the argument. The dynamic dimensions can be either a single
    integer or a list of integers.

    if `dynamic_arg_dims` is `None`, it is inferred from the type annotation
    of the `forward` method, based on the following default rules:

    - if the argument is annotated as `torch.Tensor` or
        `Optional[torch.Tensor]`, the first dimension will be
        marked as dynamic.
    - if the argument is annotated as `IntermediateTensors`, the first
        dimension of all the tensors in the intermediate tensors
        will be marked as dynamic.

    During runtime, when we actually mark dimensions of tensors,
     it depends on the value of arguments:

    - if it is a single integer (can be negative), the corresponding dimension
        of the argument will be marked as dynamic.
    - if it is `None`, ignored.
    - if it is `IntermediateTensors`, all the tensors in the intermediate
        tensors will be marked as dynamic.
    - otherwise, it will raise an error.

    NOTE: if an argument is `None`, it should always be passed as `None` during
    the lifetime of the model, otherwise, it cannot be captured as a single
    computation graph.

    `enable_if` is a function that takes a `VllmConfig` object as input and
    returns a boolean value indicating whether to compile the model or not.
    This is useful if you want to compile the model only when certain
    conditions are met.

    `mark_unbacked_dims` is a dictionary that maps argument names with a dynamic
    dim to be decorated with `mark_unbacked`.  This is useful if we would like to
    enforce that dynamo does not specialize on 0/1 values in the case of dummy input
    such as for vision model compilation

    `is_encoder` marks this module as a portion of an multimodal encoder.
    When True, the compile range upper bound is set to MAX_INT32 instead of
    max_num_batched_tokens, since encoder input shapes are unpredictable.
    This is typically used for vision encoder sub-modules in multimodal models.

    `shape_invariants` is a function that gets compiled right before forward.
    The function should have the torch._check calls that are needed to set
    the relationships between different input sizes. For example:
            torch._check(input_ids.size()[0] == inputs_embeds.size()[0])
    This enforces constraints on the symbolic shapes without hardcoding
    specific values. It is needed for some models to avoid data dependent
    errors.
    """

    def cls_decorator_helper(cls: type[_T]) -> type[_T]:
        # helper to pass `dynamic_arg_dims` to `_support_torch_compile`
        # to avoid too much indentation for `_support_torch_compile`
        if not hasattr(cls, "forward"):
            raise TypeError("decorated class should have a forward method.")
        sig = inspect.signature(cls.forward)
        inferred_dynamic_arg_dims = dynamic_arg_dims
        if inferred_dynamic_arg_dims is None:
            inferred_dynamic_arg_dims = {}
            for k, v in sig.parameters.items():
                if v.annotation in [
                    torch.Tensor,
                    torch.Tensor | None,
                    IntermediateTensors,
                    IntermediateTensors | None,
                ]:
                    inferred_dynamic_arg_dims[k] = 0

            logger.debug(
                ("Inferred dynamic dimensions for forward method of %s: %s"),
                cls,
                list(inferred_dynamic_arg_dims.keys()),
            )

        if len(inferred_dynamic_arg_dims) == 0:
            raise ValueError(
                "No dynamic dimensions found in the forward method of "
                f"{cls}. Please provide dynamic_arg_dims explicitly."
            )

        for k in inferred_dynamic_arg_dims:
            if k not in sig.parameters:
                raise ValueError(
                    f"Argument {k} not found in the forward method of {cls}"
                )
        return _support_torch_compile(
            cls,
            inferred_dynamic_arg_dims,
            mark_unbacked_dims,
            enable_if,
            is_encoder,
            shape_invariants,
        )

    if cls is not None:
        # use `support_torch_compile` as a decorator without arguments
        assert isinstance(cls, type)
        return cls_decorator_helper(cls)

    return cls_decorator_helper


def _support_torch_compile(
    cls: type[_T],
    dynamic_arg_dims: dict[str, int | list[int]],
    mark_unbacked_dims: dict[str, int | list[int]] | None = None,
    enable_if: Callable[[VllmConfig], bool] | None = None,
    is_encoder: bool = False,
    shape_invariants: Callable[..., None] = lambda *args, **kwargs: None,
) -> type[_T]:
    """
    A decorator to add support for compiling the forward method of a class.
    """
    if TorchCompileWithNoGuardsWrapper in cls.__bases__:
        # support decorating multiple times
        return cls

    # take care of method resolution order
    # make sure super().__init__ is called on the base class
    #  other than TorchCompileWithNoGuardsWrapper
    cls.__bases__ = cls.__bases__ + (TorchCompileWithNoGuardsWrapper,)

    old_init = cls.__init__

    setattr(cls, IGNORE_COMPILE_KEY, False)

    def __init__(
        self: _T,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        if vllm_config is None:
            vllm_config = get_current_vllm_config()

        # NOTE: to support multimodal models (such as encoder),
        # we may not have vllm_config so we may need to patch it
        sig = inspect.signature(old_init)
        if "vllm_config" in sig.parameters:
            kwargs["vllm_config"] = vllm_config
        if "prefix" in sig.parameters:
            kwargs["prefix"] = prefix
        old_init(self, **kwargs)

        self.vllm_config = vllm_config
        self.compilation_config = self.vllm_config.compilation_config
        enable_compile = enable_if is None or enable_if(vllm_config)
        # for CompilationMode.STOCK_TORCH_COMPILE , the upper level model runner
        # will handle the compilation, so we don't need to do anything here.
        self.do_not_compile = (
            self.compilation_config.mode
            in [CompilationMode.NONE, CompilationMode.STOCK_TORCH_COMPILE]
            or _should_ignore_torch_compile(self.__class__)
            or not enable_compile
        )
        if self.do_not_compile:
            return

        self._check_shape_invariants = shape_invariants
        compilation_counter.num_models_seen += 1
        self.compiled = False
        self._aot_runner = (
            AotRunner(vllm_config, self.forward) if envs.VLLM_USE_AOT_COMPILE else None
        )  # type: ignore[assignment]

        # Handled by monkeypatching `TorchCompileWithNoGuardsWrapper` into base class
        TorchCompileWithNoGuardsWrapper.__init__(
            self,
            compile_prefix=cls.__name__ if is_encoder else "",
            is_encoder=is_encoder,
        )

    cls.__init__ = __init__

    def _mark_dynamic_inputs(
        mod: type[_T], ds_type: DynamicShapesType, *args: Any, **kwargs: Any
    ) -> None:
        def mark_dynamic(arg: torch.Tensor, dims: list[int]) -> None:
            if ds_type == DynamicShapesType.UNBACKED:
                if is_torch_equal_or_newer("2.10.0"):
                    for dim in dims:
                        torch._dynamo.decorators.mark_unbacked(
                            arg, dim, hint_override=arg.size()[dim]
                        )
                else:
                    torch._dynamo.decorators.mark_unbacked(arg, dims)
            else:
                torch._dynamo.mark_dynamic(arg, dims)

        sig = inspect.signature(mod.__class__.forward)  # type: ignore[attr-defined]
        bound_args = sig.bind(mod, *args, **kwargs)
        bound_args.apply_defaults()
        for k, dims in dynamic_arg_dims.items():
            arg = bound_args.arguments.get(k)

            if arg is not None:
                dims = [dims] if isinstance(dims, int) else dims
                if isinstance(arg, torch.Tensor):
                    # In case dims is specified with negative indexing
                    dims = [arg.ndim + dim if dim < 0 else dim for dim in dims]
                    mark_dynamic(arg, dims)
                elif isinstance(arg, IntermediateTensors):
                    for tensor in arg.tensors.values():
                        # In case dims is specified with negative indexing
                        dims = [tensor.ndim + dim if dim < 0 else dim for dim in dims]
                        mark_dynamic(tensor, dims)
                else:
                    raise ValueError(
                        "Unsupported dynamic dimensions"
                        f" {dims} for argument {k} with type {type(arg)}."
                    )
        if mark_unbacked_dims:
            for k, dims in mark_unbacked_dims.items():
                arg = bound_args.arguments.get(k)
                if arg is not None:
                    dims = [dims] if isinstance(dims, int) else dims
                    if isinstance(arg, torch.Tensor):
                        # In case dims is specified with negative indexing
                        dims = [arg.ndim + dim if dim < 0 else dim for dim in dims]
                        if is_torch_equal_or_newer("2.10.0"):
                            for dim in dims:
                                torch._dynamo.decorators.mark_unbacked(
                                    arg, dim, hint_override=arg.size()[dim]
                                )
                        else:
                            torch._dynamo.decorators.mark_unbacked(arg, dims)

    def __call__(self: type[_T], *args: Any, **kwargs: Any) -> Any:
        # torch.compiler.is_compiling() means we are inside the compilation
        # e.g. TPU has the compilation logic in model runner, so we don't
        # need to compile the model inside.
        if self.do_not_compile or torch.compiler.is_compiling():
            return self.forward(*args, **kwargs)

        # If skip_compiled is set, bypass compiled model call. This is used e.g. for
        # enc-dec models where tensor shapes/types vary across invocations, preventing
        # the capture of a single computational graph.
        if is_forward_context_available() and get_forward_context().skip_compiled:
            return self.forward(*args, **kwargs)

        # --- AOT hot paths (A: in memory, B: disk cache hit) ---
        if self._aot_runner is not None:
            ok, out = self._aot_runner.try_run_hot(self, *args, **kwargs)
            if ok:
                return out
            ok, out = self._aot_runner.try_load_and_run(self, *args, **kwargs)
            if ok:
                self.compiled = True
                return out

        if self.compiled:
            assert (
                not envs.VLLM_USE_AOT_COMPILE
                or self.vllm_config.compilation_config.backend == "eager"
            )
            return TorchCompileWithNoGuardsWrapper.__call__(self, *args, **kwargs)  # type: ignore[arg-type]

        # This is the path for the first compilation.
        # the first compilation needs to have dynamic shapes marked
        ds_type = self.compilation_config.dynamic_shapes_config.type
        _mark_dynamic_inputs(
            self,
            ds_type,
            *args,
            **kwargs,
        )

        original_code_object = self.original_code_object()
        logger.debug("Start compiling function %s", original_code_object)

        # we do not want tp delete the original code object entries since
        # we depend on them now to look up cached compiled functions.
        # torch._dynamo.eval_frame.remove_from_cache(original_code_object)

        # collect all relevant files traced by Dynamo,
        # so that the compilation cache can trigger re-compilation
        # properly when any of these files change.

        # 1. the file containing the top-level forward function
        self.compilation_config.traced_files.add(original_code_object.co_filename)

        # 2. every time Dynamo sees a function call, it will inline
        # the function by calling InliningInstructionTranslator.inline_call_
        # we hijack this function to know all the functions called
        # during Dynamo tracing, and their corresponding files
        inline_call = InliningInstructionTranslator.inline_call_

        def patched_inline_call(self_: Any) -> Any:
            code = self_.f_code
            self.compilation_config.traced_files.add(code.co_filename)
            return inline_call(self_)

        # Disable the C++ compilation of symbolic shape guards. C++-fication
        # of symbolic shape guards can improve guard overhead. But, since
        # vllm skip guards anyways, setting this flag to False can improve
        # compile time.
        dynamo_config_patches = {}
        try:
            _ = torch._dynamo.config.enable_cpp_symbolic_shape_guards
            dynamo_config_patches["enable_cpp_symbolic_shape_guards"] = False
        except AttributeError:
            # Note: this config is not available in torch 2.6, we can skip
            # if the config doesn't exist
            logger.debug("enable_cpp_symbolic_shape_guards config not available")

        # Prepare backed_size_oblivious config patch if needed
        fx_config_patches = {}
        if ds_type == DynamicShapesType.BACKED_SIZE_OBLIVIOUS:
            fx_config_patches["backed_size_oblivious"] = True

        # Prepare inductor config patches
        # assume_32bit_indexing is only available in torch 2.10.0+
        inductor_config_patches = {}
        if is_torch_equal_or_newer("2.10.0"):
            inductor_config_patches["assume_32bit_indexing"] = (
                self.compilation_config.dynamic_shapes_config.assume_32_bit_indexing
            )

        with (
            patch.object(
                InliningInstructionTranslator, "inline_call_", patched_inline_call
            ),
            torch._dynamo.config.patch(**dynamo_config_patches),
            maybe_use_cudagraph_partition_wrapper(self.vllm_config),
            torch.fx.experimental._config.patch(**fx_config_patches),
            torch._inductor.config.patch(**inductor_config_patches),
        ):
            if self._aot_runner is not None:
                handled, output = self._aot_runner.compile_and_run(
                    self, *args, **kwargs
                )
                if handled:
                    self.compiled = True
                    return output
            with monitor_torch_compile(
                self.vllm_config,
                "torch.compile and initial profiling/warmup "
                "run together took %.2f s in total",
            ):
                output = TorchCompileWithNoGuardsWrapper.__call__(
                    self,  # type: ignore[arg-type]
                    *args,
                    **kwargs,
                )

        self.compiled = True
        return output

    cls.__call__ = __call__
    return cls


@contextlib.contextmanager
def maybe_use_cudagraph_partition_wrapper(
    vllm_config: VllmConfig,
) -> Generator[None, None, None]:
    """
    Context manager to set/unset customized cudagraph partition wrappers.

    If we're using Inductor-based graph partitioning, we currently have the
    whole `fx.Graph` before Inductor lowering and the piecewise
    splitting happens after all graph passes and fusions. Here, we add
    a custom hook for Inductor to wrap each partition with our static
    graph wrapper class to maintain more control over static graph
    capture and replay.
    """
    from vllm.config import CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    if (
        compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
        and compilation_config.use_inductor_graph_partition
    ):
        from torch._inductor.utils import CUDAGraphWrapperMetadata

        from vllm.compilation.cuda_graph import CUDAGraphOptions
        from vllm.platforms import current_platform

        static_graph_wrapper_class = resolve_obj_by_qualname(
            current_platform.get_static_graph_wrapper_cls()
        )

        def customized_cudagraph_wrapper(
            f: Callable[..., Any], metadata: CUDAGraphWrapperMetadata
        ) -> Any:
            partition_id = metadata.partition_index
            num_partitions = metadata.num_partitions
            return static_graph_wrapper_class(
                runnable=f,
                vllm_config=vllm_config,
                runtime_mode=CUDAGraphMode.PIECEWISE,
                cudagraph_options=CUDAGraphOptions(
                    debug_log_enable=partition_id == 0,
                    gc_disable=partition_id != 0,
                    weak_ref_output=partition_id == num_partitions - 1,
                ),
            )

        torch._inductor.utils.set_customized_partition_wrappers(
            customized_cudagraph_wrapper
        )

    yield

    if (
        compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
        and compilation_config.use_inductor_graph_partition
    ):
        torch._inductor.utils.set_customized_partition_wrappers(None)
