#!/usr/bin/env python3
"""Regenerate the explainer: narration (macOS say) -> timings -> frames -> mp4 (+srt, poster, gif)."""
import re, subprocess, sys, pathlib, textwrap, shutil, os
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
import numpy as np
from multiprocessing import Pool
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
import scenes as S

HERE = pathlib.Path(__file__).parent
WORK = HERE / ".work"; WORK.mkdir(exist_ok=True)
VOICE, RATE = os.environ.get("UQ_VOICE", "Samantha"), os.environ.get("UQ_RATE", "170")
PAD, GAP = 1.2, 0.35

def parse_script():
    d = _parse()
    if "vendors" in d and not S.vendor_rows(): del d["vendors"]
    return d

def _parse():
    out, cur = {}, None
    for ln in open(HERE / "script.md"):
        ln = ln.strip()
        if ln.startswith("## "): cur = ln[3:]; out[cur] = []
        elif cur and ln and not ln.startswith(("#", "<!--")):
            # split on sentence ends (keep decimals like 0.54)
            out[cur] += re.split(r"(?<=[.!?])\s+(?=[A-Z])", ln)
    return out

import functools
@functools.lru_cache(None)
def dur(p):
    p = str(p)
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))

def narrate(script):
    """returns per scene: sentences, cue start times, scene length; builds one wav per scene"""
    plan = []
    for name, sents in script.items():
        parts, cues, t = [], [], PAD
        for i, s in enumerate(sents):
            f = WORK / f"{name}_{i}.wav"
            if not f.exists():
                subprocess.check_call(["say", "-v", VOICE, "-r", RATE, "-o", str(WORK / "t.aiff"), s])
                subprocess.check_call(["ffmpeg", "-y", "-v", "error", "-i", str(WORK / "t.aiff"), "-ar", "44100", "-ac", "1", str(f)])
            d = dur(f); cues.append(t); parts.append((t, f)); t += d + GAP
        T = t + 0.9
        plan.append(dict(name=name, sents=sents, cues=cues, T=T, parts=parts))
    return plan

_fig = {}
def render(args):
    fn, name, t, cues, T, caption, path = args
    if "f" not in _fig:
        f = plt.figure(figsize=(W / 100, H / 100), dpi=100); ax = f.add_axes([0, 0, 1, 1])
        _fig["f"], _fig["ax"] = f, ax
    f, ax = _fig["f"], _fig["ax"]
    ax.clear(); ax.set_xlim(0, 192); ax.set_ylim(0, 108); ax.axis("off"); f.patch.set_facecolor(BG); ax.set_facecolor(BG)
    g = S.G(ax)
    fn(g, t, cues, T)
    if caption:
        txt = "\n".join(textwrap.wrap(caption, 95))
        ax.text(96, 3.2, txt, fontsize=17, color=FG, ha="center", va="center", linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.35", fc="#000000", ec="none", alpha=0.65))
    f.canvas.draw()
    buf = np.asarray(f.canvas.buffer_rgba())[:, :, :3]
    return buf.tobytes()

def srt_time(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"

def main():
    quick = "--quick" in sys.argv
    plan = narrate(parse_script())
    fns = dict(S.SCENES)
    total = sum(p["T"] for p in plan)
    print(f"total {total:.1f}s")
    # audio track + srt
    aud, srt, off, n = [], [], 0.0, 1
    inputs, filt = [], []
    for p in plan:
        for i, (t, f) in enumerate(p["parts"]):
            inputs += ["-i", str(f)]
            filt.append(f"[{len(filt)}:a]adelay={int((off + t) * 1000)}:all=1[a{len(filt)}]")
            end = off + (p["cues"][i + 1] if i + 1 < len(p["cues"]) else p["T"] - 0.5)
            srt.append(f"{n}\n{srt_time(off + t)} --> {srt_time(end - 0.05)}\n{p['sents'][i]}\n"); n += 1
        off += p["T"]
    open(HERE / "ultraquant_explainer.srt", "w").write("\n".join(srt))
    mix = "".join(f"[a{i}]" for i in range(len(filt)))
    subprocess.check_call(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex",
        ";".join(filt) + f";{mix}amix=inputs={len(filt)}:normalize=0[o]", "-map", "[o]", "-t", f"{total:.2f}", str(WORK / "audio.wav")])
    # frames
    out = HERE / ("ultraquant_explainer_quick.mp4" if quick else "ultraquant_explainer.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", str(WORK / "audio.wav"), "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    jobs = []
    for p in plan:
        nf = int(p["T"] * FPS)
        for k in range(nf):
            t = k / FPS
            cap = next((p["sents"][i] for i in range(len(p["sents"]) - 1, -1, -1) if t >= p["cues"][i] - 0.05), None)
            if t > p["cues"][-1] + dur(WORK / f"{p['name']}_{len(p['sents'])-1}.wav") + 0.4: cap = None
            jobs.append((fns[p["name"]], p["name"], t, p["cues"], p["T"], cap, None))
    if quick: jobs = jobs[::30]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for b in pool.imap(render, jobs, chunksize=8): ff.stdin.write(b)
    ff.stdin.close(); ff.wait()
    # poster + gif
    pt = next(j for j in jobs if j[1] == "results" and j[2] > plan[4]["cues"][4] + 1.5)
    from PIL import Image
    Image.frombytes("RGB", (W, H), render(pt[:6] + (None,))).save(HERE / "poster.png")
    if not quick:
        subprocess.check_call(["ffmpeg", "-y", "-v", "error", "-ss", "0", "-t", str(min(total, 200)), "-i", str(out), "-vf",
            "fps=4,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse", "-t", "45", str(HERE / "preview.gif")])
    print("wrote", out, f"{dur(out):.1f}s")

if __name__ == "__main__": main()
