import os, sys
import pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uqmt import attn as A, tree as T

HAS_GPU = torch.cuda.is_available() and A.HAVE_TRITON
gpu = pytest.mark.skipif(not HAS_GPU, reason="needs GPU+triton")


def test_parents_to_mask():
    assert T.parents_to_mask([-1, 0, 0, 1]).tolist() == [1, 3, 5, 11]
    assert T.parents_to_mask([-1, 0, 1, 2]).tolist() == T.chain_mask(4).tolist()
    assert T.parents_to_mask([-1, -1, -1]).tolist() == [1, 2, 4]  # independent drafts
    assert T.star_mask(4).tolist() == [1, 3, 5, 9]


def _tree(kind, K):
    if kind == "chain":
        return T.chain_mask(K)
    if kind == "binary":
        return T.parents_to_mask([-1] + [(j - 1) // 2 for j in range(1, K)])
    if kind == "star":
        return T.star_mask(K)
    g = torch.Generator().manual_seed(K)
    return T.parents_to_mask([-1] + [int(torch.randint(-1, j, (1,), generator=g)) for j in range(1, K)])


def _cache(B, T_, Hkv, D, page, lens, seed=0):
    torch.manual_seed(seed)
    k = torch.randn(B, Hkv, T_, D, device="cuda"); v = torch.randn(B, Hkv, T_, D, device="cuda")
    return A.build_paged_cache(k, v, page, seq_lens=lens, shuffle=True)


@gpu
@pytest.mark.parametrize("kind", ["chain", "binary", "star", "random"])
@pytest.mark.parametrize("K", [1, 2, 4, 7, 8, 16])
@pytest.mark.parametrize("Hq,Hkv,D", [(8, 2, 128), (4, 4, 64)])
def test_tree_vs_reference(kind, K, Hq, Hkv, D):
    lens = [333, 64 + K, 130, 1000]  # ragged incl. spec tokens on a page boundary
    c = _cache(4, 1024, Hkv, D, 64, lens)
    q = torch.randn(4, K, Hq, D, device="cuda", dtype=torch.bfloat16)
    tm = _tree(kind, K).to("cuda")
    tm = tm.expand(4, K).contiguous()
    out = T.uq_tree_decode(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens, tm)
    ref = T.tree_reference(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens, tm)
    assert (out.float() - ref).abs().max() < 2e-2


@gpu
@pytest.mark.parametrize("K", [1, 3, 16])
def test_chain_equals_causal_kernel(K):
    c = _cache(3, 700, 2, 128, 64, [700, 200, 450])
    q = torch.randn(3, K, 8, 128, device="cuda", dtype=torch.bfloat16)
    base = A.uq_decode(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens)
    masked = T.uq_tree_decode(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens, T.chain_mask(K, device="cuda"))
    unmasked = T.uq_tree_decode(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens, None)
    assert (masked.float() - base.float()).abs().max() < 1e-2
    assert torch.equal(unmasked, base)


@gpu
def test_per_batch_different_masks():
    K = 8
    c = _cache(2, 300, 2, 128, 64, [300, 280])
    q = torch.randn(2, K, 8, 128, device="cuda", dtype=torch.bfloat16)
    tm = torch.stack([_tree("binary", K), _tree("star", K)]).to("cuda")
    out = T.uq_tree_decode(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens, tm)
    ref = T.tree_reference(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens, tm)
    assert (out.float() - ref).abs().max() < 2e-2
