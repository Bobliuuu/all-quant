"""KV-cache pseudo-quantization configs used by the model-quality experiments.

Each config maps a (B, H, S, D) K or V tensor to its dequantized version and
reports the bits per value it costs (codes + scales, nothing hidden).
Rates are averaged over K and V (equal element counts), which is the analogue
of Pearl's "overall average" over GLM-5.3's NoPE and RoPE components.
"""
from __future__ import annotations

import torch

from pqrepro import baselines as BL
from pqrepro.codec import E8KVCodec

_CODECS = {}


def _codec(kind, p, D):
    k = (kind, p, D)
    if k not in _CODECS:
        _CODECS[k] = E8KVCodec(kind, p, D=D)
    return _CODECS[k]


def _pq(kind, p):
    def f(x):
        c = _codec(kind, p, x.shape[-1])
        return c.quant_dequant(x.float()).to(x.dtype), c.bpv
    return f


def _wrap(fn, **kw):
    def f(x):
        y, bpv = fn(x.float(), **kw)
        return y.to(x.dtype), bpv
    return f


def ident(x):
    return x, 16.0


# name -> (K quantizer, V quantizer)
CONFIGS = {
    "bf16": (ident, ident),
    "fp8": (_wrap(BL.fp8_e4m3, block=128), _wrap(BL.fp8_e4m3, block=128)),
    "fp8-noscale": (_wrap(BL.fp8_e4m3, block=None), _wrap(BL.fp8_e4m3, block=None)),
    "nvfp4": (_wrap(BL.nvfp4), _wrap(BL.nvfp4)),
    "int4-g32": (_wrap(BL.int_grid, bits=4, block=32), _wrap(BL.int_grid, bits=4, block=32)),
    "tq-mse-3": (_wrap(BL.turboquant_mse, bits=3), _wrap(BL.turboquant_mse, bits=3)),
    "tq-mse-2": (_wrap(BL.turboquant_mse, bits=2), _wrap(BL.turboquant_mse, bits=2)),
    # pq-repro, ball ("probable points") codebook. bits per 8-group: K / V
    "pq-3.2": (_pq("ball", 25), _pq("ball", 24)),   # 3.1875 bpv
    "pq-2.3": (_pq("ball", 18), _pq("ball", 17)),   # 2.3125 bpv
    "pq-2.1": (_pq("ball", 16), _pq("ball", 16)),   # 2.125 bpv
    # pq-repro, Voronoi (NestQuant) codebook at nearby rates
    "vor-3.2": (_pq("voronoi", 9), _pq("voronoi", 8)),  # 26/24 bits -> 3.25 bpv
    "vor-2.3": (_pq("voronoi", 5), _pq("voronoi", 4)),  # 19/16 bits -> 2.3125 bpv
    "vor-2.1": (_pq("voronoi", 4), _pq("voronoi", 4)),  # 16/16 bits -> 2.125 bpv
    # bit-split ablation at ~3.2: V gets the extra bit instead of K
    "pq-3.2-vk": (_pq("ball", 24), _pq("ball", 25)),
}

# Our addition (not from Pearl): K mean-centering. Subtracting one per-(layer, head)
# vector mu from every key changes every logit of a query by the same q.mu, so the
# softmax is unchanged: the decode kernel never needs mu. mu = mean of the first
# 256 prefill keys (computed once, online). "-kvc" also centers V (o += mu_v at the end).
for _n in ["pq-3.2", "pq-2.3", "pq-2.1", "vor-3.2", "vor-2.3", "vor-2.1", "tq-mse-3", "tq-mse-2", "nvfp4"]:
    CONFIGS[_n + "-kc"] = CONFIGS[_n] + ({"kcenter": 256},)
for _n in ["pq-3.2", "pq-2.3", "pq-2.1"]:
    CONFIGS[_n + "-kvc"] = CONFIGS[_n] + ({"kcenter": 256, "vcenter": 256},)

