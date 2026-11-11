"""CPU check: is uqmt.quant.quantize(rule='paper', c=0.156) bit-identical to the vendored upstream vLLM
reference encoder (cuda/compat/upstream/reference.py)?  Run: python -I cuda/compat/paper_rule_cpu_check.py"""
import sys, os, types, importlib.util, json, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m
for p in ("vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.ops", "vllm.v1.attention.ops.ultraquant"):
    sys.modules[p] = types.ModuleType(p)
load("vllm.v1.attention.ops.ultraquant.format", os.path.join(HERE, "upstream", "format.py"))
UR = load("vllm.v1.attention.ops.ultraquant.reference", os.path.join(HERE, "upstream", "reference.py"))
sys.path.insert(0, ROOT)
from uqmt import quant as UQ
def canon_negzero(pk):
    """map FP4 negative-zero nibble 0x8 -> 0x0 (upstream bucketize never emits -0)."""
    lo, hi = pk & 0xF, pk >> 4
    lo = torch.where(lo == 8, torch.zeros_like(lo), lo); hi = torch.where(hi == 8, torch.zeros_like(hi), hi)
    return (lo | (hi << 4)).to(torch.uint8)
torch.manual_seed(0)
out = {}
for name, gen in [("gauss", lambda n, D: torch.randn(n, D)),
                  ("lognormal_groups", lambda n, D: torch.randn(n, D) * torch.exp(0.7 * torch.randn(n, D // 32, 1).expand(n, D // 32, 32).reshape(n, D))),
                  ("bf16_gauss_x4", lambda n, D: (torch.randn(n, D) * 4).bfloat16().float()),
                  ("with_zero_groups", lambda n, D: torch.randn(n, D) * (torch.rand(n, D // 32, 1) > 0.2).float().expand(n, D // 32, 32).reshape(n, D))]:
    for D in (128, 256):
        x = gen(20000, D)
        enc = UR.ultraquant_encode(x, rotate=False)
        pk, sc = UQ.quantize(x, "paper", 0.156)
        pkz = canon_negzero(pk)
        pk_c, sc_c = UQ.quantize(x, "ceil")
        out[f"{name}_D{D}"] = dict(
            paper_packed_equal=float((pk == enc.codes_packed).float().mean()), paper_scale_equal=float((sc == enc.scale_bytes).float().mean()),
            paper_packed_equal_after_negzero_canon=float((pkz == enc.codes_packed).float().mean()), paper_packed_bit_identical_after_negzero_canon=bool(torch.equal(pkz, enc.codes_packed)), paper_packed_bit_identical=bool(torch.equal(pk, enc.codes_packed)), paper_scale_bit_identical=bool(torch.equal(sc, enc.scale_bytes)),
            ceil_scale_equal=float((sc_c == enc.scale_bytes).float().mean()))
        print(name, D, out[f"{name}_D{D}"])
json.dump(out, open(os.path.join(ROOT, "results", "cuda_compat_paper_rule_cpu.json"), "w"), indent=1)
