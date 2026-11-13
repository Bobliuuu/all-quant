"""E8 lattice primitives for pq-repro (PearlQuant-style E8 KV codec).

Coordinates: E8 = D8 u (D8 + 1/2), i.e. vectors in Z^8 or (Z+1/2)^8 with even
coordinate sum. Internally we often use "doubled" integer coordinates y = 2x:
all-even or all-odd integers with sum(y) = 0 (mod 4).

The nearest-point rule is the Conway & Sloane D8 / D8+1/2 decoder (SPLAG ch. 20).
HyperQuant (MIT, github.com/moonmath-ai/HyperQuant, hyperquant/lattice.py) has an
equivalent E8 rounding; this file is an independent implementation of the same
textbook algorithm.
"""
from __future__ import annotations

import torch

# Conway & Sloane generator for E8 (rows = basis vectors, det 1).
GEN = torch.tensor(
    [
        [2, 0, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, -1, 1, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, -1, 1, 0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ],
    dtype=torch.float64,
)
GEN_INV = torch.linalg.inv(GEN)


def _nearest_d8(x: torch.Tensor) -> torch.Tensor:
    f = torch.round(x)
    s = f.sum(-1, keepdim=True)
    odd = (torch.remainder(s, 2) != 0)
    # flip the coordinate with the largest rounding error toward x
    err = x - f
    i = err.abs().argmax(-1, keepdim=True)
    ei = torch.gather(err, -1, i)
    step = torch.where(ei >= 0, torch.ones_like(ei), -torch.ones_like(ei))
    fi = torch.gather(f, -1, i) + step
    g = f.scatter(-1, i, fi)
    return torch.where(odd, g, f)


def nearest_e8(x: torch.Tensor) -> torch.Tensor:
    """Nearest E8 point to each row of x (..., 8). Returns same dtype."""
    a = _nearest_d8(x)
    b = _nearest_d8(x - 0.5) + 0.5
    da = ((x - a) ** 2).sum(-1, keepdim=True)
    db = ((x - b) ** 2).sum(-1, keepdim=True)
    return torch.where(da <= db, a, b)


def to_coords(y: torch.Tensor) -> torch.Tensor:
    """E8 point -> integer coordinates c with y = c @ GEN."""
    return torch.round(y.double() @ GEN_INV.to(y.device)).long()


def from_coords(c: torch.Tensor) -> torch.Tensor:
    return c.double() @ GEN.to(c.device)
