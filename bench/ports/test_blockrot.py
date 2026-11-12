"""correctness of the fused block-rotation ports vs the original amd-experiments quantisers and a torch attention on the dequantised cache."""
import sys, os, torch; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockrot_common as c, torch.nn.functional as F
dev = "cuda"; torch.manual_seed(0)
for m in ("planar", "iso", "rotor"):
    for bits in (3, 4):
        p = c.make(m, bits, dev); ref = c.bq.make_quantizer(m, bits=bits, head_dim=128, device=dev)
        orth = (p.Rf @ p.Rf.T - torch.eye(128, device=dev)).abs().max().item()
        x = (torch.randn(4096, 128, device=dev) * torch.rand(1, 128, device=dev) * 3).half()
        o = ref.decompress(ref.compress(x), x.shape).float(); f = p.fake_quant(x).float()
        # kernel quantiser vs fake_quant
        P, NR = p._quant(x); p.state = [P, NR, P, NR]; kd = p.dequant(0, (1, 1, 4096)).reshape(4096, 128)
        cs = lambda a, b: F.cosine_similarity(a, b, dim=-1).mean().item()
        print(f"{m}{bits} orth_err={orth:.2e} cos(orig_roundtrip,x)={cs(o, x.float()):.4f} cos(port_fake,orig)={cs(f, o):.4f} cos(kernel_dq,fake)={cs(kd, f):.5f} maxabs(kernel_dq-fake)={(kd-f).abs().max().item():.3e}", flush=True)
# decode kernel vs torch attention on dequantised cache
p = c.make("iso", 3, dev); B, S = 2, 2048
kf = torch.randn(B, 8, S, 128, device=dev).half(); vf = torch.randn_like(kf); p.write(kf, vf)
Kd, Vd = p.dequant(0, (B, 8, S)), None
p.state = [p.state[0], p.state[1], p.state[2], p.state[3]]
Vd = p.dequant(1, (B, 8, S)) if False else None
P, NR, VP, VN = p.state; p2 = c.make("iso", 3, dev); p2.state = [VP, VN, VP, VN]; Vd = p2.dequant(0, (B, 8, S))
for K in (1, 2, 8):
    q = torch.randn(B, K, 64, 128, device=dev).bfloat16(); o = p.decode(q).float()
    G = 8; qq = q.float().reshape(B, K, 8, G, 128); sc = torch.einsum("bkhgd,bhsd->bhgks", qq, Kd) * 128 ** -0.5
    lim = S - K + 1 + torch.arange(K, device=dev); sc = sc.masked_fill(torch.arange(S, device=dev)[None, :] >= lim[:, None], float("-inf"))
    r = torch.einsum("bhgks,bhsd->bkhgd", sc.softmax(-1), Vd).reshape(B, K, 64, 128)
    print("decode K", K, "cos vs torch-on-dequant", F.cosine_similarity(o.reshape(-1, 128), r.reshape(-1, 128), dim=-1).min().item(), "maxabs", (o - r).abs().max().item())
