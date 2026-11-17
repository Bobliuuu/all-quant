"""Item 9: bytes-vs-instructions roofline for fused decode-GEMV, per chip, validated against modal_kernels output.

  python research/weight-reencode/roofline.py results/H100.json results/B200.json results/RTX-PRO-6000.json

Model. A batch-1 decode GEMV over W weights at b bits/weight (side info included) takes
    t = max( W * b / 8 / BW ,  W * I / R )
BW = achievable DRAM bandwidth, I = issued SASS instructions per weight per lane (decode + per-weight share of the
loop: loads, mma, addressing), R = lane-instruction issue rate = SMs * 4 schedulers * 32 lanes * clock.
A codec stops paying when W*I/R exceeds W*b/8/BW, i.e. when I > R * b / (8 * BW) = the ALU budget per weight.
I is measured, not assumed: the L2-resident run (every m-tile reads the same 8 tiles) is compute bound, so
I_eff = R / (decoded weights per second). That folds in latency/dependency stalls, which a static count misses.
"""
import json
import sys

SPEC = {  # SMs, boost clock (GHz), DRAM BW spec (TB/s)
    "H100": dict(sms=132, ghz=1.98, bw=3.35, mem="HBM3"),
    "B200": dict(sms=148, ghz=1.965, bw=8.0, mem="HBM3e"),
    "RTX-PRO-6000": dict(sms=188, ghz=2.617, bw=1.792, mem="GDDR7"),
}


def issue_rate(gpu):
    s = SPEC[gpu]
    return s["sms"] * 4 * 32 * s["ghz"] * 1e9  # lane-instructions / s


def budget(gpu, bpw, bw=None):
    """ALU budget: lane instructions per weight that fit under the memory time at bpw."""
    bw = (bw or SPEC[gpu]["bw"]) * 1e12
    return issue_rate(gpu) * bpw / (8 * bw)


def main(files):
    for f in files:
        d = json.load(open(f))
        gpu = d["gpu"]
        R = issue_rate(gpu)
        print(f"\n## {gpu}: issue {R / 1e12:.1f} T lane-instr/s, spec BW {SPEC[gpu]['bw']} TB/s, d2d copy "
              f"{d['copy_gbs'] / 1e3:.2f} TB/s (r+w)")
        print("budget (instr/weight) at 16/4.25/3.25/2.25/1.6 bpw:",
              ", ".join(f"{budget(gpu, b):.1f}" for b in (16, 4.25, 3.25, 2.25, 1.6)))
        rows = d["rows"]
        l2 = {}
        for r in rows:
            if r.get("mode") == "l2_resident":
                k = r["codec"]
                l2[k] = max(l2.get(k, 0), r["gweights_per_s"])
        best = {}
        for r in rows:
            if r.get("mode") or r["shape"] != "gate_up" or r["B"] != 1:
                continue
            k = r["codec"]
            if k not in best or r["us"] < best[k]["us"]:
                best[k] = r
        print("| codec | bpw | I_eff (instr/w) | budget at bpw | predicted us | measured us | eff GB/s (bf16-equiv) |")
        print("|---|---|---|---|---|---|---|")
        W = 24576 * 4096
        bw = SPEC[gpu]["bw"] * 1e12
        for k, r in sorted(best.items(), key=lambda kv: kv[1]["us"]):
            bpw = r.get("bpw", 16.0)
            ieff = R / (l2[k] * 1e9) if k in l2 else float("nan")
            t_mem = W * bpw / 8 / bw * 1e6
            t_alu = W * ieff / R * 1e6 if k in l2 else 0
            print(f"| {k} | {bpw:.2f} | {ieff:.1f} | {budget(gpu, bpw):.1f} | {max(t_mem, t_alu):.1f} | {r['us']:.1f} | "
                  f"{r['eff_gbs']:.0f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
