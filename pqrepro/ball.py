"""Ball-truncated E8 codebook with leader-based indexing ("probable points").

Pearl's blog: "the lattice has infinitely many points, but we only assign indices
to probable lattice points under the Gaussian i.i.d. distribution". For an iid
Gaussian, a lattice point's probability decreases with its norm, so the 2^b most
probable points are the 2^b points of smallest norm. That is our reading; Pearl
does not say how indices are assigned. We choose the classic leader
decomposition used in lattice VQ (e.g. the RE8 codebooks of AMR-WB+/EVS):

  every E8 point = signed permutation of an "absolute leader"
  (sorted, non-negative coordinates). Index layout, leaders sorted by norm:

  idx = offset[leader] + perm_rank * nsign[leader] + sign_rank

Doubled coordinates y = 2x are used throughout: E8 points are all-even or
all-odd integer vectors with sum(y) = 0 mod 4.
 * all-even leaders: every sign pattern is valid, nsign = 2^(#nonzero)
 * all-odd leaders: half of the 2^8 sign patterns are valid, nsign = 128
   (signs of positions 0..6 are free, position 7 restores sum = 0 mod 4)
perm_rank is the lexicographic rank (ascending |value|) of the multiset
permutation; it has the closed-form update used in rank()/unrank().
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .e8 import nearest_e8

FACT = [math.factorial(i) for i in range(9)]


def _gen_leaders(max_norm2: int):
    """All absolute leaders (doubled coords, descending) with sum(y^2) <= max_norm2."""
    out = []

    def rec(prefix, maxv, rem, parity):
        if len(prefix) == 8:
            out.append(tuple(prefix))
            return
        v = maxv
        while v >= 0:
            if v % 2 == parity and v * v <= rem:
                rec(prefix + [v], v, rem - v * v, parity)
            v -= 1

    top = int(math.isqrt(max_norm2))
    for parity in (0, 1):
        rec([], top, max_norm2, parity)
    res = []
    for a in out:
        if a[0] % 2 == 0 and sum(a) % 4 != 0:
            continue
        res.append(a)
    return res


def _orbit(a):
    mult = {}
    for v in a:
        mult[v] = mult.get(v, 0) + 1
    nperm = FACT[8]
    for c in mult.values():
        nperm //= FACT[c]
    if a[0] % 2 == 1:
        nsign = 128
    else:
        nsign = 2 ** sum(1 for v in a if v != 0)
    return nperm, nsign


@dataclass
class BallTables:
    bits: int
    leaders: np.ndarray  # (L, 8) int, descending abs values (doubled coords)
    norm2: np.ndarray  # (L,) doubled norm^2
    nperm: np.ndarray
    nsign: np.ndarray
    offset: np.ndarray  # (L,) int64 start index
    size: np.ndarray  # (L,) number of indices actually used (last one partial)
    keys: np.ndarray  # sorted lookup keys
    key_to_leader: np.ndarray
    max_abs: int

    @property
    def L(self):
        return len(self.leaders)


def _key(a: np.ndarray) -> np.ndarray:
    k = np.zeros(a.shape[:-1], dtype=np.int64)
    for i in range(8):
        k = k * 64 + a[..., i].astype(np.int64)
    return k


@lru_cache(maxsize=None)
def ball_tables(bits: int) -> BallTables:
    M = 1 << bits
    n2 = 8
    while True:
        lead = _gen_leaders(n2)
        tot = sum(np.prod(_orbit(a)) for a in lead)
        if tot >= M:
            break
        n2 *= 2
    lead.sort(key=lambda a: (sum(v * v for v in a), a))
    rows, nperm, nsign, off, size = [], [], [], [], []
    acc = 0
    for a in lead:
        p, s = _orbit(a)
        if acc >= M:
            break
        rows.append(a)
        nperm.append(p)
        nsign.append(s)
        off.append(acc)
        size.append(min(p * s, M - acc))
        acc += p * s
    leaders = np.array(rows, dtype=np.int64)
    keys = _key(leaders)
    order = np.argsort(keys)
    return BallTables(
        bits=bits,
        leaders=leaders,
        norm2=(leaders ** 2).sum(1),
        nperm=np.array(nperm, dtype=np.int64),
        nsign=np.array(nsign, dtype=np.int64),
        offset=np.array(off, dtype=np.int64),
        size=np.array(size, dtype=np.int64),
        keys=keys[order],
        key_to_leader=order,
        max_abs=int(leaders.max()),
    )


def rank(y2: torch.Tensor, T: BallTables) -> torch.Tensor:
    """Doubled-coordinate E8 points (N,8) long -> index, or -1 if not in codebook."""
    dev = y2.device
    a = y2.abs()
    srt = torch.sort(a, dim=-1, descending=True).values
    keys = torch.zeros(a.shape[0], dtype=torch.long, device=dev)
    for i in range(8):
        keys = keys * 64 + srt[:, i].clamp(max=63)
    tk = torch.as_tensor(T.keys, device=dev)
    pos = torch.searchsorted(tk, keys).clamp(max=len(T.keys) - 1)
    found = (tk[pos] == keys) & (srt.max(-1).values <= T.max_abs)
    lid = torch.as_tensor(T.key_to_leader, device=dev)[pos]
    V = T.max_abs + 2
    cnt = torch.zeros(a.shape[0], V, dtype=torch.long, device=dev)
    cnt.scatter_add_(1, a.clamp(max=V - 1), torch.ones_like(a))
    fact = torch.as_tensor(FACT, dtype=torch.long, device=dev)
    P = fact[cnt].prod(-1)
    r = torch.zeros_like(keys)
    ar = torch.arange(V, device=dev)
    for i in range(8):
        n = 8 - i
        v = a[:, i].clamp(max=V - 1)
        less = (cnt * (ar[None] < v[:, None])).sum(-1)
        r = r + fact[n - 1] * less // P
        cv = cnt.gather(1, v[:, None]).squeeze(1)
        P = P // cv.clamp(min=1)
        cnt.scatter_add_(1, v[:, None], -torch.ones_like(v[:, None]))
    neg = (y2 < 0).long()
    odd = (a[:, 0] % 2 == 1)
    # odd: bits of positions 0..6
    w7 = (1 << torch.arange(7, device=dev)).long()
    s_odd = (neg[:, :7] * w7).sum(-1)
    # even: bits over nonzero positions in order
    nz = (a != 0).long()
    j = torch.cumsum(nz, -1) - nz
    s_even = (neg * nz * (1 << j)).sum(-1)
    s = torch.where(odd, s_odd, s_even)
    nsign = torch.as_tensor(T.nsign, device=dev)[lid]
    local = r * nsign + s
    idx = torch.as_tensor(T.offset, device=dev)[lid] + local
    ok = found & (local < torch.as_tensor(T.size, device=dev)[lid])
    return torch.where(ok, idx, torch.full_like(idx, -1))


def unrank(idx: torch.Tensor, T: BallTables) -> torch.Tensor:
    """index (N,) long -> doubled-coordinate E8 points (N,8) long."""
    dev = idx.device
    off = torch.as_tensor(T.offset, device=dev)
    lid = torch.searchsorted(off, idx, right=True) - 1
    local = idx - off[lid]
    nsign = torch.as_tensor(T.nsign, device=dev)[lid]
    r = local // nsign
    s = local % nsign
    lead = torch.as_tensor(T.leaders, device=dev)[lid]
    V = T.max_abs + 2
    cnt = torch.zeros(idx.shape[0], V, dtype=torch.long, device=dev)
    cnt.scatter_add_(1, lead, torch.ones_like(lead))
    fact = torch.as_tensor(FACT, dtype=torch.long, device=dev)
    P = fact[cnt].prod(-1)
    out = torch.zeros(idx.shape[0], 8, dtype=torch.long, device=dev)
    for i in range(8):
        n = 8 - i
        t = r * P // fact[n - 1]
        cum = torch.cumsum(cnt, -1)
        v = (cum <= t[:, None]).sum(-1)  # smallest v with cum(v) > t
        cless = torch.where(v > 0, cum.gather(1, (v - 1).clamp(min=0)[:, None]).squeeze(1), torch.zeros_like(v))
        r = r - fact[n - 1] * cless // P
        cv = cnt.gather(1, v[:, None]).squeeze(1)
        P = P // cv
        cnt.scatter_add_(1, v[:, None], -torch.ones_like(v[:, None]))
        out[:, i] = v
    odd = (lead[:, 0] % 2 == 1)
    w7 = (1 << torch.arange(7, device=dev)).long()
    neg7 = (s[:, None] // w7) % 2
    y_odd = out.clone()
    y_odd[:, :7] = torch.where(neg7 == 1, -out[:, :7], out[:, :7])
    part = y_odd[:, :7].sum(-1) + out[:, 7]
    y_odd[:, 7] = torch.where(part % 4 == 0, out[:, 7], -out[:, 7])
    nz = (out != 0).long()
    j = torch.cumsum(nz, -1) - nz
    negE = ((s[:, None] >> j) & 1) * nz
    y_even = torch.where(negE == 1, -out, out)
    return torch.where(odd[:, None], y_odd, y_even)


def quantize(u: torch.Tensor, T: BallTables, return_idx=False):
    """Quantize scaled groups u (N,8) to the ball codebook. Returns points in x-coords.

    Nearest E8 point; if it falls outside the codebook (overload), shrink the
    input radially until its nearest point is a codeword (approximate projection).
    """
    y = nearest_e8(u.double())
    y2 = torch.round(2 * y).long()
    idx = rank(y2, T)
    bad = idx < 0
    # start the radial shrink at the codebook's outer radius, then step down 3%
    rmax = math.sqrt(T.norm2[-1] / 4.0)
    g0 = (rmax / u.double().norm(dim=-1).clamp(min=1e-30)).clamp(max=1.0)
    k = 0
    while bad.any():
        g = g0[bad][:, None] * (0.97 ** k)
        k += 1
        ub = u[bad].double() * g
        yb = nearest_e8(ub)
        y2b = torch.round(2 * yb).long()
        ib = rank(y2b, T)
        sel = torch.nonzero(bad).squeeze(1)
        good = ib >= 0
        y2[sel[good]] = y2b[good]
        idx[sel[good]] = ib[good]
        bad[sel[good]] = False
    out = y2.to(u.dtype) / 2
    return (out, idx) if return_idx else out
