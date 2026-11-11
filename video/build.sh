#!/bin/sh
# Regenerates video/ultraquant_explainer.mp4 (+ .srt, poster.png, preview.gif) from results/*.json. Needs ffmpeg, python3 (numpy, matplotlib, pillow), macOS say.
set -e
cd "$(dirname "$0")"
python3 build.py "$@"
