# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parametrized mathematical-property tests for all standard RoPE variants.

Instantiation uses the public get_rope() factory — the same path that model
code uses — so the tests also exercise parameter dispatch.

Six properties are verified for every included variant:
  1. cos² + sin² = norm²          (norm=1 for most; norm=mscale for YaRN/DeepSeek)
  2. position 0 → cos = norm, sin = 0   (cache level)
  3. _compute_inv_freq is strictly monotonically decreasing
  4. Shape consistency: forward_native output shapes match input shapes
  5. Zero-position no-rotation: forward_native(position=0, q, k) returns norm*q, norm*k
  6. Relative position property: q_m · k_n == q_{m+d} · k_{n+d}

Excluded variants (require separate tests):
  - DualChunkRotaryEmbedding        : five distinct caches; needs a CUDA device at init
  - FourierRotaryEmbedding          : 3-D cache (max_pos, num_kv_heads, kv_size*2);
                                      lazy update
  - Phi3LongRoPEScaledRotaryEmbedding : cos_sin_cache is a two-segment concat with
                                        distinct mscales per segment
  - Llama4VisionRotaryEmbedding     : uses complex-valued cache (torch.view_as_complex),
                                      not the standard (cos‖sin) concat format
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.rotary_embedding import get_rope

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

HEAD_SIZE = 64
MAX_POSITION = 128
BASE = 10000.0
DTYPE = torch.float32
SCALING_FACTOR = 2.0
NUM_HEADS = 2
NUM_TOKENS = 4

_MROPE_SECTION = [8, 12, 12]  # sums to HEAD_SIZE // 2 = 32


# ---------------------------------------------------------------------------
# Per-variant test configuration
# ---------------------------------------------------------------------------


def _make_qk_standard(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard shape: (num_tokens, num_heads * head_size)."""
    q = torch.randn(num_tokens, NUM_HEADS * HEAD_SIZE, dtype=DTYPE)
    k = torch.randn(num_tokens, NUM_HEADS * HEAD_SIZE, dtype=DTYPE)
    return q, k


def _make_qk_deepseek(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek forward_native expects (batch=1, seq_len, num_heads, head_size)."""
    q = torch.randn(1, num_tokens, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    k = torch.randn(1, num_tokens, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    return q, k


def _pos_1d(num_tokens: int, pos: int) -> torch.Tensor:
    """1-D positions for standard RoPE variants."""
    return torch.full((num_tokens,), pos, dtype=torch.long)


def _pos_2d(num_tokens: int, pos: int) -> torch.Tensor:
    """(1, num_tokens) positions for DeepSeek which expects (batch, seq_len)."""
    return torch.full((1, num_tokens), pos, dtype=torch.long)


def _pos_3row(num_tokens: int, pos: int) -> torch.Tensor:
    """(3, num_tokens) positions for MRoPE variants (T/H/W)."""
    return torch.full((3, num_tokens), pos, dtype=torch.long)


def _pos_4row(num_tokens: int, pos: int) -> torch.Tensor:
    """(4, num_tokens) positions for XDRoPE variants (P/W/H/T)."""
    return torch.full((4, num_tokens), pos, dtype=torch.long)


@dataclass
class RopeTestConfig:
    name: str
    # rope_parameters dict passed to get_rope()
    rope_params: dict[str, Any] = field(default_factory=dict)
    # Some variants multiply cos/sin by mscale; provide a callable that reads
    # the scalar norm from the instantiated module (default: always 1.0).
    get_norm: Callable[[nn.Module], float] = field(default=lambda _: 1.0)
    # Set to False for variants that override _compute_inv_freq with a
    # non-standard formula (DeepSeek uses YaRN blending per dimension).
    test_inv_freq_monotone: bool = True
    # Returns a positions tensor of the right shape for forward_native().
    # Signature: (num_tokens: int, position: int) -> Tensor
    make_positions: Callable[[int, int], torch.Tensor] = field(default=_pos_1d)
    # XDRoPE forward_native asserts key is not None; set True to always pass key.
    requires_key: bool = False
    # Factory for (q, k) tensors; variants with non-standard input shapes override this.
    make_qk: Callable[[int], tuple[torch.Tensor, torch.Tensor]] = field(
        default=_make_qk_standard
    )


ALL_CONFIGS = [
    RopeTestConfig(
        name="RotaryEmbedding",
        rope_params={},
    ),
    RopeTestConfig(
        name="LinearScalingRotaryEmbedding",
        rope_params={"rope_type": "linear", "factor": SCALING_FACTOR},
    ),
    RopeTestConfig(
        name="DynamicNTKScalingRotaryEmbedding",
        rope_params={"rope_type": "dynamic", "factor": SCALING_FACTOR},
    ),
    RopeTestConfig(
        name="DynamicNTKAlphaRotaryEmbedding",
        rope_params={"rope_type": "dynamic", "alpha": SCALING_FACTOR},
    ),
    RopeTestConfig(
        name="NTKScalingRotaryEmbedding",
        rope_params={"rope_type": "ntk", "factor": SCALING_FACTOR},
    ),
    RopeTestConfig(
        name="YaRNScalingRotaryEmbedding",
        rope_params={
            "rope_type": "yarn",
            "factor": SCALING_FACTOR,
            "original_max_position_embeddings": MAX_POSITION,
        },
        get_norm=lambda rope: rope.mscale,
    ),
    RopeTestConfig(
        name="Llama3RotaryEmbedding",
        rope_params={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    ),
    RopeTestConfig(
        name="MRotaryEmbedding",
        rope_params={"mrope_section": _MROPE_SECTION},
        make_positions=_pos_3row,
        requires_key=True,
    ),
    RopeTestConfig(
        name="MRotaryEmbeddingInterleaved",
        rope_params={
            "rope_type": "openpangu",
            "mrope_section": _MROPE_SECTION,
            "mrope_interleaved": True,
        },
        make_positions=_pos_3row,
        requires_key=True,
    ),
    RopeTestConfig(
        name="XDRotaryEmbedding",
        rope_params={
            "rope_type": "xdrope",
            "alpha": 1.0,
            "xdrope_section": [8, 8, 8, 8],  # sums to HEAD_SIZE // 2 = 32
        },
        make_positions=_pos_4row,
        requires_key=True,
    ),
    RopeTestConfig(
        name="DeepseekScalingRotaryEmbedding",
        rope_params={
            "rope_type": "deepseek_yarn",
            "factor": SCALING_FACTOR,
            "original_max_position_embeddings": MAX_POSITION,
        },
        get_norm=lambda rope: rope.mscale,
        # DeepSeek overrides _compute_inv_freq with YaRN-style per-dim blending;
        # the blended result is not guaranteed to be strictly monotone.
        test_inv_freq_monotone=False,
        # forward_native requires positions (batch, seq) and
        # q/k (batch, seq, heads, dim)
        make_positions=_pos_2d,
        requires_key=True,
        make_qk=_make_qk_deepseek,
    ),
]


_ROPE_CACHE: dict[str, nn.Module] = {}


def _make_rope(cfg: RopeTestConfig) -> nn.Module:
    return get_rope(
        head_size=HEAD_SIZE,
        max_position=MAX_POSITION,
        is_neox_style=True,
        rope_parameters={"rope_theta": BASE, **cfg.rope_params}
        if cfg.rope_params
        else None,
        dtype=DTYPE,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.name)
def test_cos_sin_identity(default_vllm_config, cfg: RopeTestConfig):
    """cos² + sin² should equal norm² for every position and dimension."""
    rope = _make_rope(cfg)
    norm = cfg.get_norm(rope)
    cache = rope.cos_sin_cache  # (rows, rotary_dim)
    cos, sin = cache.chunk(2, dim=-1)
    identity = cos**2 + sin**2
    expected = torch.full_like(identity, norm**2)
    torch.testing.assert_close(identity, expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.name)
def test_position_zero(default_vllm_config, cfg: RopeTestConfig):
    """At position 0 all angles are 0: cos = norm, sin = 0."""
    rope = _make_rope(cfg)
    norm = cfg.get_norm(rope)
    cache = rope.cos_sin_cache
    cos0, sin0 = cache[0].chunk(2, dim=-1)
    torch.testing.assert_close(cos0, torch.full_like(cos0, norm), atol=1e-5, rtol=0)
    torch.testing.assert_close(sin0, torch.zeros_like(sin0), atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "cfg",
    [c for c in ALL_CONFIGS if c.test_inv_freq_monotone],
    ids=lambda c: c.name,
)
def test_inv_freq_monotone(default_vllm_config, cfg: RopeTestConfig):
    """inv_freq should be strictly decreasing: low dims rotate fast, high dims slow."""
    rope = _make_rope(cfg)
    inv_freq = rope._compute_inv_freq(BASE)  # (rotary_dim // 2,)
    diffs = inv_freq[1:] - inv_freq[:-1]
    assert (diffs < 0).all(), (
        f"{cfg.name}: inv_freq is not strictly decreasing. "
        f"First non-decreasing diff at index "
        f"{(diffs >= 0).nonzero(as_tuple=True)[0][0].item()}"
    )


# ---------------------------------------------------------------------------
# forward_native-based tests (CPU, no GPU required)
# ---------------------------------------------------------------------------


def _forward(
    rope: nn.Module,
    cfg: RopeTestConfig,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out, k_out = rope.forward_native(positions, q.clone(), k.clone())
    return q_out, k_out


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.name)
def test_shape_consistency(default_vllm_config, cfg: RopeTestConfig):
    """forward_native must not change the shape of query or key tensors."""
    rope = _make_rope(cfg)
    q, k = cfg.make_qk(NUM_TOKENS)
    positions = cfg.make_positions(NUM_TOKENS, 5)
    q_out, k_out = _forward(rope, cfg, positions, q, k)
    assert q_out.shape == q.shape, (
        f"{cfg.name}: q shape changed {q.shape} -> {q_out.shape}"
    )
    assert k_out.shape == k.shape, (
        f"{cfg.name}: k shape changed {k.shape} -> {k_out.shape}"
    )


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.name)
def test_zero_position_no_rotation(default_vllm_config, cfg: RopeTestConfig):
    """At position 0 the rotation angle is 0: forward_native returns norm*q, norm*k.

    For standard variants norm=1 (identity).
    For YaRN/DeepSeek norm=mscale (uniform scale).
    """
    rope = _make_rope(cfg)
    norm = cfg.get_norm(rope)
    q, k = cfg.make_qk(NUM_TOKENS)
    positions = cfg.make_positions(NUM_TOKENS, 0)
    q_out, k_out = _forward(rope, cfg, positions, q, k)
    torch.testing.assert_close(q_out, norm * q, atol=1e-5, rtol=0)
    torch.testing.assert_close(k_out, norm * k, atol=1e-5, rtol=0)


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.name)
def test_relative_position_property(default_vllm_config, cfg: RopeTestConfig):
    """The RoPE relative-position invariant: q_m · k_n == q_{m+d} · k_{n+d}.

    Uses uniform positions across all modality rows so section-based variants
    (MRoPE, XDRoPE) reduce to standard RoPE and the property still holds.
    """
    rope = _make_rope(cfg)
    q, k = cfg.make_qk(1)  # single token for a clean dot-product check

    m, n, d = 5, 10, 7

    pos_m = cfg.make_positions(1, m)
    pos_n = cfg.make_positions(1, n)
    pos_md = cfg.make_positions(1, m + d)
    pos_nd = cfg.make_positions(1, n + d)

    q_m, _ = _forward(rope, cfg, pos_m, q, k)
    _, k_n = _forward(rope, cfg, pos_n, q, k)
    q_md, _ = _forward(rope, cfg, pos_md, q, k)
    _, k_nd = _forward(rope, cfg, pos_nd, q, k)

    dot_mn = (q_m * k_n).sum()
    dot_mdnd = (q_md * k_nd).sum()
    torch.testing.assert_close(dot_mn, dot_mdnd, atol=1e-3, rtol=1e-3)
