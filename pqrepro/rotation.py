"""Random rotation: random sign flip followed by a normalized Walsh-Hadamard
transform (a "randomized Hadamard transform", as in QuaRot/NestQuant/HyperQuant).
Pearl only says "after a random rotation"; the Hadamard choice is ours.
"""
from __future__ import annotations

import torch


def hadamard(n: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)


def signs(n: int, seed: int = 0x5051) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()


class RHT:
    def __init__(self, n: int, seed: int = 0x5051):
        self.n = n
        self.s = signs(n, seed)
        self.H = hadamard(n)

    def fwd(self, x):  # x (..., n)
        return (x * self.s.to(x)) @ self.H.to(x)

    def inv(self, y):
        return (y @ self.H.to(y).T) * self.s.to(y)
