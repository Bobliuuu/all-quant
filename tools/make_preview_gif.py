#!/usr/bin/env python3
"""Rebuild video/preview.gif as a slideshow: one settled frame per scene, each held HOLD seconds.

video-readme-maker squeezes the whole explainer into a ~14 s GIF, which at 20 scenes flashes each one
for under half a second on the GitHub README. Run this after `vrm.py video README.spec.yml --silent`:
it reads the scene timings that run left in video/.render/data.json and the rendered mp4.
"""
import json, pathlib, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "video/.render/data.json"
MP4 = ROOT / "video/ultraquant_explainer.mp4"
OUT = ROOT / "video/preview.gif"
HOLD, AT, WIDTH = 2.5, 0.85, 720  # seconds per scene, point in each scene to sample (fraction), GIF width


def main():
    if not DATA.exists() or not MP4.exists():
        sys.exit(f"need {DATA.relative_to(ROOT)} and {MP4.relative_to(ROOT)}: run vrm.py video README.spec.yml --silent first")
    scenes = json.load(open(DATA, encoding="utf-8"))["scenes"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        lines = []
        for i, s in enumerate(scenes):
            frame = tmp / f"s{i:02d}.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s['start'] + s['dur'] * AT:.2f}", "-i", str(MP4),
                            "-frames:v", "1", str(frame)], check=True)
            lines += [f"file '{frame.as_posix()}'", f"duration {HOLD}"]
        lines.append(lines[-2])  # concat demuxer: repeat the last file so its duration is honoured
        (tmp / "list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"), "-vf",
                        f"fps=2,scale={WIDTH}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        "-loop", "0", str(OUT)], check=True)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(scenes)} scenes x {HOLD}s, {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
