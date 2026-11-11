"""Shared style + data loading. All numbers come from results/*.json."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results"
W, H, FPS = 1920, 1080, 30
BG, FG, MUTED, GRID = "#0f1419", "#e6edf3", "#8b98a5", "#26303a"
BASE, OURS, MI, B3 = "#f59e0b", "#2dd4bf", "#60a5fa", "#c084fc"

def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None

def rows(name):
    d = load(name)
    return {r["K"]: r for r in d} if d else {}
