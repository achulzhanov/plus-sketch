#!/usr/bin/env python3
"""
verify_psk.py -- read a .psk back and prove it's correct.

Two jobs:
  1. Validate export_fixed.py before any C is written.
  2. Serve as the executable spec for core/'s loader. Whatever this does,
     the C must do -- same order, same packing, same fixed-point maths.

    python verify_psk.py --psk=../models/plus_sketch_q4.psk
    python verify_psk.py --psk=../models/plus_sketch_q4.psk --draw=cat
"""

import argparse
import struct

import numpy as np
import torch

from model import ModelArgs, Transformer

MAGIC = b"PSK1"
HEADER_BYTES = 128
FX_ONE = 65536.0


def unpack_int4(buf, n):
    """Low nibble first, two's complement. Must match export_fixed.pack_int4."""
    a = np.frombuffer(buf, dtype=np.uint8)
    lo = (a & 0xF).astype(np.int8)
    hi = (a >> 4).astype(np.int8)
    lo[lo > 7] -= 16
    hi[hi > 7] -= 16
    out = np.empty(len(a) * 2, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out[:n].astype(np.float64)


class Reader:
    def __init__(self, path):
        self.buf = open(path, "rb").read()
        self.p = 0

    def take(self, n):
        b = self.buf[self.p:self.p + n]
        assert len(b) == n, "file truncated"
        self.p += n
        return b

    def arr(self, dtype, n):
        return np.frombuffer(self.take(np.dtype(dtype).itemsize * n), dtype=dtype)


def read_psk(path):
    r = Reader(path)
    head = r.take(HEADER_BYTES)
    magic, ver, bits, group, dim, hidden, n_layers, n_heads, n_kv, vocab, \
        seq, shared, n_draw, n_jump = struct.unpack("<4siiiiiiiiiiiii", head[:56])
    assert magic == MAGIC, f"bad magic {magic!r}"
    cfg = dict(bits=bits, group=group, dim=dim, hidden_dim=hidden,
               n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv,
               vocab_size=vocab, max_seq_len=seq, shared=bool(shared))
    print(f"PSK v{ver}: {bits}-bit, group {group}")
    for k, v in cfg.items():
        if k not in ("bits", "group"):
            print(f"  {k}={v}")

    draw_b = r.arr("<i2", n_draw * 2).reshape(n_draw, 2)
    jump_b = r.arr("<i2", n_jump * 2).reshape(n_jump, 2)
    print(f"  codebook: {n_draw} draw, {n_jump} jump")

    sd = {}
    for i in range(n_layers):
        sd[f"layers.{i}.attention_norm.weight"] = r.arr("<i4", dim) / FX_ONE
        sd[f"layers.{i}.ffn_norm.weight"] = r.arr("<i4", dim) / FX_ONE
    sd["norm.weight"] = r.arr("<i4", dim) / FX_ONE

    kv_dim = (dim // n_heads) * n_kv
    shapes = [("tok_embeddings.weight", (vocab, dim))]
    for i in range(n_layers):
        p = f"layers.{i}."
        shapes += [
            (p + "attention.wq.weight", (dim, dim)),
            (p + "attention.wk.weight", (kv_dim, dim)),
            (p + "attention.wv.weight", (kv_dim, dim)),
            (p + "attention.wo.weight", (dim, dim)),
            (p + "feed_forward.w1.weight", (hidden, dim)),
            (p + "feed_forward.w2.weight", (dim, hidden)),
            (p + "feed_forward.w3.weight", (hidden, dim)),
        ]
    if not shared:
        shapes.append(("output.weight", (vocab, dim)))

    for name, shape in shapes:
        n = int(np.prod(shape))
        if bits == 32:
            sd[name] = r.arr("<f4", n).astype(np.float64).reshape(shape)
            continue
        n_groups = -(-n // group)
        scales = r.arr("<i4", n_groups).astype(np.float64) / FX_ONE
        if bits == 8:
            q = r.arr(np.int8, n).astype(np.float64)
        else:
            q = unpack_int4(r.take(-(-n // 2)), n)
        sd[name] = (q * np.repeat(scales, group)[:n]).reshape(shape)

    assert r.p == len(r.buf), f"{len(r.buf) - r.p} bytes unread -- format mismatch"
    print(f"  consumed all {r.p:,} bytes")
    return cfg, sd, draw_b, jump_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psk", default="../models/plus_sketch_q4.psk")
    ap.add_argument("--checkpoint", default="out/ckpt.pt")
    ap.add_argument("--draw", default="")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.2)
    a = ap.parse_args()

    cfg, sd, draw_b, jump_b = read_psk(a.psk)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    ref = ck["model"]
    for k in list(ref):
        if k.startswith("_orig_mod."):
            ref[k[len("_orig_mod."):]] = ref.pop(k)

    print("\nper-tensor relative error vs fp32 checkpoint:")
    worst = ("", 0.0)
    for name in sorted(sd):
        if name not in ref:
            continue
        a_, b_ = sd[name], ref[name].float().numpy()
        e = np.abs(a_ - b_).mean() / max(np.abs(b_).mean(), 1e-12)
        if e > worst[1]:
            worst = (name, e)
    print(f"  worst: {worst[0]}  {100*worst[1]:.2f}%")
    norm_err = max(
        np.abs(sd[k] - ref[k].float().numpy()).max()
        for k in sd if "norm" in k and k in ref)
    print(f"  norms (16.16, unquantized): max abs error {norm_err:.2e}")

    if not a.draw:
        print("\nformat OK. pass --draw=cat to sample from the reconstructed model.")
        return

    from sketchdata import decode
    from sample import generate, normalize, categories
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = ModelArgs(dim=cfg["dim"], n_layers=cfg["n_layers"],
                     n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"],
                     vocab_size=cfg["vocab_size"], hidden_dim=cfg["hidden_dim"],
                     max_seq_len=cfg["max_seq_len"], dropout=0.0)
    m = Transformer(args)
    state = {k: torch.tensor(v, dtype=torch.float32) for k, v in sd.items()}
    if cfg["shared"]:
        state["output.weight"] = state["tok_embeddings.weight"]
    m.load_state_dict(state, strict=False)
    m.eval()

    cats = categories()
    cid = cats.index(a.draw)
    torch.manual_seed(1337)
    fig, axes = plt.subplots(1, a.n, figsize=(3 * a.n, 3), squeeze=False)
    for ax in axes[0]:
        toks = generate(m, cid, a.temperature, cfg["max_seq_len"], "cpu")
        for s in normalize(decode(toks, draw_b.astype(np.int32),
                                  jump_b.astype(np.int32))):
            xs, ys = zip(*s)
            ax.plot(xs, ys, "k-", lw=2)
        ax.invert_yaxis(); ax.axis("equal"); ax.axis("off")
    bits = cfg["bits"]
    out = f"psk_{a.draw}_int{bits}.png"
    fig.suptitle(f"{a.draw} — from {a.psk} (int{bits}, t={a.temperature})")
    plt.savefig(out, dpi=80, bbox_inches="tight")
    print(f"\nwrote {out} -- compare against q_{a.draw}_int{bits}_g{cfg['group']}.png")


if __name__ == "__main__":
    main()
