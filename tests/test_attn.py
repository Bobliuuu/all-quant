"""Tests for uqmt. CPU tests always run; GPU tests skip cleanly without a GPU/triton."""
import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import uqmt  # noqa: E402
from uqmt import attn as A  # noqa: E402

HAS_GPU = torch.cuda.is_available() and A.HAVE_TRITON
gpu = pytest.mark.skipif(not HAS_GPU, reason="needs a GPU and triton")
DEV = "cuda"


# ------------------------------------------------------------------ CPU only
def test_import_without_gpu():
    assert callable(uqmt.uq_decode) and callable(uqmt.quantize_fp4) and callable(uqmt.reference)


def test_fp4_grid_roundtrip_exact():
    # every representable value (x scale) must survive quantize -> dequantize exactly
    grid = torch.tensor(A.FP4_VALUES)
    vals = torch.cat([grid, -grid[1:]])
    x = torch.zeros(2, 32)
    x[0, : vals.numel()] = vals
    x[0, 31] = 6.0  # pins scale to 1.0
    x[1] = x[0] * 8.0
    p, s = uqmt.quantize_fp4(x)
    assert p.dtype == torch.uint8 and p.shape == (2, 16) and s.shape == (2, 1)
    assert torch.equal(uqmt.dequantize_fp4(p, s), x)
    assert int(s[0, 0]) == 127 and int(s[1, 0]) == 130


@pytest.mark.parametrize("D", [64, 128])
def test_quantize_error_bound(D):
    torch.manual_seed(0)
    x = torch.randn(64, 4, D)
    p, s = uqmt.quantize_fp4(x)
    y = uqmt.dequantize_fp4(p, s)
    # e2m1 worst-case step is 2 (between 4 and 6) in units of scale; scale <= amax/3 .. amax/6
    err = (x - y).abs().reshape(64, 4, D // 32, 32).amax(-1)
    amax = x.abs().reshape(64, 4, D // 32, 32).amax(-1)
    assert (err <= amax / 3 + 1e-6).all()


def test_negative_zero_codes_decode_to_zero():
    p = torch.tensor([[0x88, 0x08, 0x80, 0x00] * 4], dtype=torch.uint8)  # 16 bytes = 32 values
    s = torch.full((1, 1), 127, dtype=torch.uint8)
    y = uqmt.dequantize_fp4(p, s)
    assert (y == 0).all()


def test_scale_extremes_cpu():
    x = torch.tensor([[1e-30] * 32, [1e30] * 32])
    p, s = uqmt.quantize_fp4(x)
    y = uqmt.dequantize_fp4(p, s)
    assert torch.isfinite(y).all()
    assert torch.allclose(y, x, rtol=0.35)


def test_build_paged_cache_cpu_shuffle_matches_dense():
    torch.manual_seed(1)
    k = torch.randn(3, 2, 50, 64)
    v = torch.randn(3, 2, 50, 64)
    for shuffle in (False, True):
        c = uqmt.build_paged_cache(k, v, 16, seq_lens=[50, 33, 5], shuffle=shuffle)
        assert c.block_table.shape == (3, 4) and c.seq_lens.tolist() == [50, 33, 5]
        # gather b=1 logical pages and compare against direct dequant
        pg = c.block_table[1].long()
        kd = uqmt.dequantize_fp4(c.kc[pg], c.ks[pg]).permute(1, 0, 2, 3).reshape(2, -1, 64)[:, :50]
        kq = uqmt.dequantize_fp4(*uqmt.quantize_fp4(k[1]))
        assert torch.equal(kd, kq)


def test_reference_fully_masked_rows_are_zero():
    c = uqmt.build_paged_cache(torch.randn(1, 1, 3, 64), torch.randn(1, 1, 3, 64), 16)
    q = torch.randn(1, 4, 2, 64).bfloat16()  # K=4 > seq_len=3 -> row 0 has lim 0
    out = uqmt.reference(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens)
    assert (out[0, 0] == 0).all() and out[0, 1:].abs().sum() > 0


def test_auto_nsplit_cpu():
    f = uqmt.auto_nsplit
    assert f(8, 8, 512) == 16
    assert f(1, 8, 4096) == 64 and f(1, 8, 4096) & 63 == 0
    assert f(8, 8, 1) == 1 and f(8, 8, 3) == 2          # never more splits than pages
    assert f(64, 8, 512) in (2, 4)
    for B, H, P in itertools.product([1, 3, 32], [1, 8], [1, 7, 100, 2048]):
        n = f(B, H, P)
        assert n & (n - 1) == 0 and 1 <= n <= 64


# ------------------------------------------------------------------ GPU
def _make(B, HQ, HKV, D, PAGE, seq_lens, K, seed=0, T=None, kscale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    T = T or max(seq_lens)
    k = torch.randn(B, HKV, T, D, device=DEV, generator=g) * kscale
    v = torch.randn(B, HKV, T, D, device=DEV, generator=g)
    c = uqmt.build_paged_cache(k, v, PAGE, seq_lens=seq_lens, shuffle=True, generator=g)
    q = torch.randn(B, K, HQ, D, device=DEV, generator=g).bfloat16()
    return q, c


def _check(q, c, atol=3e-2, **kw):
    out = c.decode(q, **kw).float()
    ref = uqmt.reference(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=atol, rtol=3e-2)
    return out


@gpu
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("PAGE", [16, 32, 64, 128])
def test_dims_and_pages(D, PAGE):
    q, c = _make(3, 16, 4, D, PAGE, [700, 200, 513], K=4)
    _check(q, c)


@gpu
@pytest.mark.parametrize("G", [1, 2, 4, 7, 8, 16])
@pytest.mark.parametrize("K", [1, 3, 8])
def test_gqa_and_k(G, K):
    HKV = 2
    q, c = _make(2, HKV * G, HKV, 128, 64, [333, 1000], K=K)
    _check(q, c)


@gpu
@pytest.mark.parametrize("K", [1, 2, 4, 8, 16])
def test_k_up_to_16(K):
    q, c = _make(2, 16, 2, 128, 64, [900, 77], K=K)
    _check(q, c)


@gpu
def test_ragged_and_short_sequences():
    # shorter than one page, exactly one page, page+1, seq_len < K, seq_len == K, ragged long
    lens = [5, 64, 65, 3, 8, 2000]
    q, c = _make(len(lens), 8, 2, 128, 64, lens, K=8)
    out = _check(q, c)
    # seq_len=3 < K=8: queries j with 3-8+j+1 <= 0 (j<=4) are fully masked -> exactly zero
    assert (out[3, :5] == 0).all()
    assert out[3, 5:].abs().sum() > 0


@gpu
def test_empty_sequence_and_empty_trailing_splits():
    # seq_len 0 -> zeros; tiny seqs inside a table with many pages -> most splits are empty
    q, c = _make(3, 8, 2, 128, 64, [0, 1, 300], K=1, T=4096)
    out = _check(q, c, nsplit=32)
    assert (out[0] == 0).all()
    q, c = _make(2, 8, 2, 128, 64, [10, 70], K=2, T=2048)
    for ns in (1, 2, 8, 64):
        _check(q, c, nsplit=ns)


@gpu
def test_nsplit_invariance():
    q, c = _make(2, 16, 2, 128, 32, [1500, 999], K=4)
    ref = c.decode(q, nsplit=1).float()
    for ns in (2, 4, 16, 64):
        torch.testing.assert_close(c.decode(q, nsplit=ns).float(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(c.decode(q).float(), ref, atol=2e-2, rtol=2e-2)  # auto


@gpu
def test_negative_zero_codes_in_kernel():
    q, c = _make(2, 8, 2, 128, 64, [200, 130], K=2)
    # flip every zero nibble (0x0) to negative zero (0x8): values must be unchanged
    def flip(x):
        lo, hi = x & 0xF, x >> 4
        lo = torch.where(lo == 0, torch.full_like(lo, 8), lo)
        hi = torch.where(hi == 0, torch.full_like(hi, 8), hi)
        return lo | (hi << 4)
    c2 = uqmt.PagedFP4Cache(flip(c.kc), c.ks, flip(c.vc), c.vs, c.block_table, c.seq_lens)
    assert (flip(c.kc) != c.kc).any()
    torch.testing.assert_close(c2.decode(q).float(), c.decode(q).float(), atol=0, rtol=0)
    _check(q, c2)


@gpu
@pytest.mark.parametrize("kscale", [2.0 ** -20, 2.0 ** 20])
def test_scale_extremes(kscale):
    # large/small K magnitude -> extreme scales + saturated/uniform softmax; must stay finite & match
    q, c = _make(2, 8, 2, 128, 64, [300, 129], K=2, kscale=kscale)
    _check(q, c, atol=5e-2)


@gpu
def test_baseline_matches_fused():
    q, c = _make(2, 16, 2, 128, 64, [400, 71], K=4)
    f = c.decode(q, fused=True).float()
    u = c.decode(q, fused=False).float()
    torch.testing.assert_close(f, u, atol=2e-2, rtol=2e-2)


@gpu
def test_int64_inputs_and_dtype_checks():
    q, c = _make(2, 8, 2, 128, 64, [100, 90], K=2)
    out = uqmt.uq_decode(q, c.kc, c.ks, c.vc, c.vs, c.block_table.long(), c.seq_lens.long())
    torch.testing.assert_close(out, c.decode(q))
    with pytest.raises(AssertionError):
        c.decode(q.float())


@gpu
def test_workspace_and_cuda_graph():
    q, c = _make(4, 16, 4, 128, 64, [900, 17, 513, 64], K=4)
    ns = 8
    ws = uqmt.UQWorkspace.create(4, 4, 16, 4, 128, ns)
    ref = c.decode(q, nsplit=ns).clone()
    out = c.decode(q, workspace=ws)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    # graph capture: no host syncs / allocations inside uq_decode
    c.decode(q, workspace=ws)  # warm up (compile)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gout = c.decode(q, workspace=ws)
    q.copy_(torch.randn_like(q.float()).bfloat16())     # new data, same address
    c.seq_lens.copy_(torch.tensor([800, 30, 400, 64], dtype=torch.int32))  # ragged lens change in place
    g.replay()
    torch.cuda.synchronize()
    expect = uqmt.reference(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens)
    torch.testing.assert_close(gout.float(), expect, atol=3e-2, rtol=3e-2)


@gpu
def test_fp8k_rejects_non128(monkeypatch):
    monkeypatch.setenv("UQ_FP8K", "1")
    q, c = _make(1, 8, 2, 64, 64, [100], K=1)
    with pytest.raises(ValueError):
        c.decode(q)


# ------------------------------------------------------------ adapter (CPU)
def test_adapter_eligibility_and_append_cpu():
    from uqmt import vllm_adapter as VA
    q = torch.randn(8, 4, 64).bfloat16()
    ok, why, K = VA.eligible(q, torch.tensor([0, 4, 8]), 64)
    assert (ok, why, K) == (True, "ok", 4)
    assert VA.eligible(q, torch.tensor([0, 3, 8]), 64)[1] == "non_uniform_k"
    assert VA.eligible(q.float(), torch.tensor([0, 4, 8]), 64)[1] == "q_dtype"
    assert VA.eligible(q, torch.tensor([0, 4, 8]), 48)[1] == "page_size"
    assert VA.eligible(q, torch.tensor([0, 4, 8]), 64, sliding_window=128)[1] == "sliding_window"
    # append_kv writes the same bytes as build_paged_cache
    P, H, D = 16, 2, 64
    kc = torch.zeros(3, H, P, D // 2, dtype=torch.uint8); vc = kc.clone()
    ks = torch.zeros(3, H, P, D // 32, dtype=torch.uint8); vs = ks.clone()
    key = torch.randn(5, H, D); val = torch.randn(5, H, D)
    slots = torch.tensor([16, 17, 18, 32, 33])
    VA.append_kv(kc, ks, vc, vs, key, val, slots)
    ref_k, ref_s = uqmt.quantize_fp4(key)
    assert torch.equal(kc[1, :, 0], ref_k[0]) and torch.equal(ks[2, :, 1], ref_s[4])
    assert (kc[0] == 0).all()


@gpu
def test_adapter_verify_matches_decode():
    from uqmt import vllm_adapter as VA
    q, c = _make(3, 16, 2, 128, 64, [300, 200, 90], K=4)
    out = VA.uq_verify_attention(q.reshape(12, 16, 128), c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens, 4)
    torch.testing.assert_close(out.reshape(3, 4, 16, 128), c.decode(q), atol=0, rtol=0)
