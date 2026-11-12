"""Launch-parameter sweep of the SpectralQuant decode kernel (B8 ctx32768, K=1 and 8, cold cache)."""
import sys, os, json, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "ports")); sys.path.insert(0, os.path.join(HERE, "..", "ports", "spectralquant"))
import torch, sq_triton as sq
from common import *
import port as P
kf, vf, qs = make_kv(8, 32768, "llm")
p = P.make_port(DEV); p.calibrate(kf[:1, :, :4096], vf[:1, :, :4096]); p.write(kf, vf)
rows = []
for K in (1, 8):
    q = torch.stack([qs(64 * K).reshape(K, 64, D) for _ in range(8)]).bfloat16()
    for ns, bn, nw in itertools.product((16, 32, 64), (32, 64, 128), (4, 8)):
        try:
            p.nsplit, p.BN, p.nw = ns, bn, nw
            ms = timeit(lambda: p.decode(q), 10, cold=True)
        except Exception as e: ms = None; print("fail", ns, bn, nw, str(e)[:60])
        rows.append(dict(K=K, nsplit=ns, BN=bn, num_warps=nw, ms=ms)); print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
