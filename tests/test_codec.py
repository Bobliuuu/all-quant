"""Quick self-checks: python -m pytest tests/ (or python tests/test_codec.py)."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pqrepro import ball as B  # noqa: E402
from pqrepro import voronoi as V  # noqa: E402
from pqrepro.codec import E8KVCodec  # noqa: E402
from pqrepro.e8 import nearest_e8  # noqa: E402


def test_theta_series():
    T = B.ball_tables(16)
    y = B.unrank(torch.arange(1 << 16), T)
    n2 = (y ** 2).sum(-1) // 4
    assert [(n2 == k).sum().item() for k in (0, 2, 4, 6, 8)] == [1, 240, 2160, 6720, 17520]


def test_rank_unrank_roundtrip():
    for b in (16, 24, 29):
        T = B.ball_tables(b)
        idx = torch.randint(0, 1 << b, (50000,))
        assert (B.rank(B.unrank(idx, T), T) == idx).all()


def test_nearest_e8_mse():
    x = torch.randn(200000, 8, dtype=torch.float64) * 4
    mse = ((x - nearest_e8(x)) ** 2).mean().item()
    assert abs(mse - 0.0717) < 0.002  # G(E8) * vol^(2/8), vol = 1


def test_voronoi_roundtrip():
    for q in (4, 7):
        c = torch.randint(0, q, (5000, 8))
        y = V.decode_coords(c, q)
        assert (V.encode_coords(y, q) == c).all()
        assert (V.unpack(V.pack(c, q), q) == c).all()


def test_codec_rate_and_quality():
    x = torch.randn(1000, 128, dtype=torch.float64)
    c = E8KVCodec("ball", 24)
    assert c.bpv == 24 / 8 + 16 / 128
    idx, s = c.encode(x)
    assert idx.max().item() < (1 << 24) and idx.min().item() >= 0
    d = ((c.decode(idx, s, torch.float64) - x) ** 2).mean().item()
    assert c.bpv + 0.5 * math.log2(d) < 0.5  # within 0.5 wasted bits of Shannon


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
