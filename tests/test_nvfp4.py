"""Tests for uqmt.nvfp4 (NVFP4 scale path).  CPU tests always run; GPU tests skip without GPU/triton."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uqmt import attn as A  # noqa: E402
from uqmt import nvfp4 as N  # noqa: E402

HAS_GPU = torch.cuda.is_available() and A.HAVE_TRITON
gpu = pytest.mark.skipif(not HAS_GPU, reason="needs a GPU and triton")
DEV = "cuda"


def test_e4m3_bit_trick_exact_all_bytes():
    # the kernel decode: fp16 bits (b&0x7F)<<7, times 2**8, must equal e4m3fn for all 127 finite positive bytes
    b = torch.arange(0, 127, dtype=torch.int32)
    got = ((b << 7).to(torch.int16)).view(torch.float16).float() * 256.0
    want = b.to(torch.uint8).view(torch.float8_e4m3fn).float()
    assert torch.equal(got, want)


def test_quantize_matches_nvfp4_spec():
    torch.manual_seed(0)
    x = torch.randn(32, 4, 128) * 3
    p, s, ts = N.quantize_nvfp4_bytes(x)
    assert p.shape == (32, 4, 64) and s.shape == (32, 4, 8) and s.dtype == torch.uint8
    assert abs(ts - float(x.abs().max()) / (6 * 448)) < 1e-9
    assert (s & 0x80 == 0).all() and (s != 0x7F).all()
    y = N.dequantize_nvfp4_bytes(p, s, ts)
    err = (x - y).abs().reshape(32, 4, 8, 16).amax(-1)
    amax = x.abs().reshape(32, 4, 8, 16).amax(-1)
    assert (err <= amax / 3 * 1.07 + 1e-6).all()   # fp4 step bound + e4m3 scale rounding slack


def _mk(B, T, HKV, D, seed=0, shuffle=True, scale=1.0, dev=DEV):
    g = torch.Generator(device=dev).manual_seed(seed)
    k = (torch.randn(B, HKV, T, D, device=dev, generator=g) * scale).to(torch.bfloat16)
    v = (torch.randn(B, HKV, T, D, device=dev, generator=g) * scale).to(torch.bfloat16)
    return k, v


def _check(B, K, HQ, HKV, D, T, seq_lens=None, PAGE=64, nsplit=None, fused=True, scale=1.0, tol=2e-2):
    k, v = _mk(B, T, HKV, D, scale=scale)
    c = N.build_paged_cache_nvfp4(k, v, PAGE, seq_lens, shuffle=True)
    g = torch.Generator(device=DEV).manual_seed(1)
    q = torch.randn(B, K, HQ, D, device=DEV, generator=g).to(torch.bfloat16)
    out = c.decode(q, nsplit=nsplit, fused=fused).float()
    ref = N.reference_nvfp4(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens, c.k_ts, c.v_ts)
    rel = (out - ref).norm() / ref.norm().clamp_min(1e-9)
    assert rel < tol, float(rel)
    return out


@gpu
@pytest.mark.parametrize("K", list(range(1, 17)))
@pytest.mark.parametrize("D", [64, 128])
def test_k_sweep(K, D):
    _check(2, K, 8, 2, D, 300)


@gpu
@pytest.mark.parametrize("G", [1, 2, 4, 8])
def test_gqa(G):
    _check(2, 4, 4 * G, 4, 128, 257)


@gpu
def test_ragged_and_short():
    _check(5, 4, 8, 2, 128, 700, seq_lens=[700, 4, 65, 129, 333])
    _check(3, 8, 8, 2, 64, 200, seq_lens=[8, 9, 200])


@gpu
def test_empty_and_fully_masked_rows():
    out = _check(2, 4, 8, 2, 128, 128, seq_lens=[2, 128])   # seq 0 < K: rows with limit<=0 are zero
    assert torch.isfinite(out).all()


@gpu
@pytest.mark.parametrize("scale", [1e-3, 1.0, 300.0])
def test_tensor_scale_ranges(scale):
    _check(2, 4, 8, 2, 128, 256, scale=scale)


@gpu
def test_nsplit_and_baseline_invariance():
    k, v = _mk(2, 512, 2, 128)
    c = N.build_paged_cache_nvfp4(k, v, 64)
    q = torch.randn(2, 8, 8, 128, device=DEV).to(torch.bfloat16)
    o1 = c.decode(q, nsplit=1).float()
    for ns in (2, 4, 8):
        assert (c.decode(q, nsplit=ns).float() - o1).abs().max() < 2e-2
    assert (c.decode(q, fused=False, nsplit=4).float() - o1).abs().max() < 2e-2


@gpu
def test_all_e4m3_scale_bytes_in_kernel():
    # inject every finite positive scale byte and compare with torch decode
    D, T = 128, 64
    k, v = _mk(1, T, 1, D)
    c = N.build_paged_cache_nvfp4(k, v, 64)
    sb = torch.arange(0, 127, dtype=torch.uint8, device=DEV)
    c.ks = sb.repeat(64 * 8 // 127 + 1)[: 64 * 8].reshape(1, 1, 64, 8).contiguous()
    c.vs = c.ks.clone()
    q = torch.randn(1, 1, 1, D, device=DEV).to(torch.bfloat16)
    out = c.decode(q).float()
    ref = N.reference_nvfp4(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens, c.k_ts, c.v_ts)
    assert (out - ref).norm() / ref.norm() < 3e-2
