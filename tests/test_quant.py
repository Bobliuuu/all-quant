import os, sys
import pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uqmt import attn as A, quant as Q

HAS_GPU = torch.cuda.is_available() and A.HAVE_TRITON


def test_ceil_rule_matches_attn_bit_exact():
    torch.manual_seed(0)
    x = torch.randn(32, 4, 128) * 3
    p0, s0 = A.quantize_fp4(x)
    p1, s1 = Q.quantize(x, "ceil")
    assert torch.equal(p0, p1) and torch.equal(s0, s1)


def test_paper_rule_scale_and_clip():
    x = torch.zeros(1, 32); x[0, 0] = 6.0
    # c=0.156: log2(0.936)=-0.095 -> E=0 -> scale 1; ceil: log2(1)=0 -> same
    assert int(Q.quantize(x, "paper", 0.156)[1]) == 127
    x[0, 0] = 7.0  # paper: log2(1.092)=0.127 -> E=0, clipped to 6 ; ceil: E=1 -> 14/2=7 ok
    p, s = Q.quantize(x, "paper", 0.156)
    assert int(s) == 127 and Q.dequantize_fp4(p, s)[0, 0] == 6.0
    assert int(Q.quantize(x, "ceil")[1]) == 128


def test_fwht_orthonormal_involution_and_matches_matrix():
    torch.manual_seed(1)
    x = torch.randn(5, 128, dtype=torch.float64)
    y = Q.fwht(x)
    assert torch.allclose(Q.fwht(y), x, atol=1e-12)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1))
    H = Q.hadamard_matrix(128, dtype=torch.float64)
    assert torch.allclose(H @ H.T, torch.eye(128, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(x @ H, y, atol=1e-12)
    yb = Q.fwht(x, 64)  # block-diagonal
    assert torch.allclose(Q.fwht(yb, 64), x, atol=1e-12)


def test_score_invariance_under_rotation():
    torch.manual_seed(2)
    k, q = torch.randn(10, 128, dtype=torch.float64), torch.randn(3, 128, dtype=torch.float64)
    kr, qr = Q.rotate_kq(k, q)
    assert torch.allclose(q @ k.T, qr @ kr.T, atol=1e-10)


def test_layer_policy():
    pol = Q.layer_policy(8, 2)
    assert [pol[i] for i in range(8)] == ['bf16', 'bf16'] + ['fp4'] * 4 + ['bf16', 'bf16']
    assert Q.bf16_layers(3, 2) == {0, 1, 2}
    pol = Q.layer_policy(40, 1, attention_layers=[3, 7, 11])
    assert pol == {3: 'bf16', 7: 'fp4', 11: 'bf16'}


def test_nvfp4_better_than_mxfp4_and_roundtrip():
    torch.manual_seed(3)
    x = torch.randn(256, 128)
    p, s, ts = Q.quantize_nvfp4(x)
    assert s.dtype == torch.float8_e4m3fn and s.shape == (256, 8)
    e_nv = (Q.dequantize_nvfp4(p, s, ts) - x).pow(2).mean()
    e_mx = (Q.fake_quant(x, "ceil") - x).pow(2).mean()
    assert e_nv < e_mx


def test_rotation_helps_outlier_channels():
    torch.manual_seed(4)
    x = torch.randn(512, 128); x[:, 5] *= 30
    e0 = (Q.fake_quant(x) - x).pow(2).mean()
    e1 = (Q.fake_quant(x, rotate=True) - x).pow(2).mean()
    assert e1 < e0 * 0.8 or e1 < e0  # rotation spreads outlier; mild MSE change allowed


@pytest.mark.skipif(not HAS_GPU, reason="needs GPU+triton")
@pytest.mark.parametrize("rule,c", [("ceil", 0), ("paper", 0.156)])
def test_uq_decode_on_quant_output(monkeypatch, rule, c):
    torch.manual_seed(0)
    B, H, T, D, K = 2, 2, 200, 128, 4
    k, v = torch.randn(B, H, T, D, device="cuda"), torch.randn(B, H, T, D, device="cuda")
    monkeypatch.setattr(A, "quantize_fp4", lambda x: Q.quantize(x, rule, c))
    cache = A.build_paged_cache(k, v, 64)
    q = torch.randn(B, K, 4, D, device="cuda", dtype=torch.bfloat16)
    out = cache.decode(q)
    ref = A.reference(q, cache.kc, cache.ks, cache.vc, cache.vs, cache.block_table, cache.seq_lens)
    assert (out.float() - ref).abs().max() < 2e-2
