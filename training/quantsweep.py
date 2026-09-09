#!/usr/bin/env python3
"""
quantsweep.py -- how much does quantization hurt, before writing any C?

Loads the fp32 checkpoint, quantizes every weight tensor and immediately
dequantizes it in memory, then draws. Nothing is written to disk in a
quantized format -- this only measures damage, so that when the fixed-point
C port produces garbage you know whether it's the quantization or your code.

    python quantsweep.py --checkpoint=out/ckpt.pt --category=cat
    python quantsweep.py --checkpoint=out/ckpt.pt --category=bicycle \\
        --temperatures=0.15,0.25,0.35

Writes one PNG per (bits, group) with n samples per temperature, plus a
greedy (t=0) row for a deterministic comparison against the fp32 reference.
"""

import argparse
import copy

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketchdata import load_books, decode
from sample import load_model, generate, normalize, categories


def fake_quant(w, bits, group):
    """Symmetric per-group round-trip: quantize then dequantize in place."""
    shape = w.shape
    flat = w.reshape(-1)
    pad = (-len(flat)) % group
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    g = flat.reshape(-1, group)
    qmax = 2 ** (bits - 1) - 1
    scale = g.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(g / scale).clamp(-qmax - 1, qmax)
    out = (q * scale).reshape(-1)
    return out[:len(flat) - pad if pad else len(flat)].reshape(shape)


def quantize_model(model, bits, group, skip_embeddings=False):
    m = copy.deepcopy(model)
    n_q = n_skip = 0
    err = []
    for name, p in m.named_parameters():
        # norms are tiny (2*dim each) and very sensitive -- never quantize
        if p.ndim < 2 or "norm" in name:
            n_skip += 1
            continue
        if skip_embeddings and ("tok_embeddings" in name or "output" in name):
            n_skip += 1
            continue
        q = fake_quant(p.data.float(), bits, group)
        err.append(((q - p.data).abs().mean() / p.data.abs().mean()).item())
        p.data = q
        n_q += 1
    return m, n_q, n_skip, float(np.mean(err)) if err else 0.0


def draw_grid(model, cat_id, temps, n, draw_b, jump_b, device, max_new, path, title):
    rows, labels = [], []
    for t in temps:
        for i in range(n):
            toks = generate(model, cat_id, t, max_new, device)
            rows.append(normalize(decode(toks, draw_b, jump_b)))
            labels.append(f"t={t}" if i == 0 else "")
    cols = n
    r = len(rows) // cols
    fig, axes = plt.subplots(r, cols, figsize=(2.4 * cols, 2.4 * r), squeeze=False)
    for ax, strokes, lab in zip(axes.flat, rows, labels):
        for s in strokes:
            xs, ys = zip(*s)
            ax.plot(xs, ys, "k-", lw=2)
        ax.invert_yaxis(); ax.axis("equal"); ax.axis("off")
        if lab:
            ax.set_ylabel(lab)
            ax.text(-0.1, 0.5, lab, transform=ax.transAxes,
                    rotation=90, va="center", fontsize=9)
    fig.suptitle(title, fontsize=11)
    plt.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="out/ckpt.pt")
    p.add_argument("--category", default="cat")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--temperatures", default="0.0,0.15,0.25,0.35")
    p.add_argument("--configs", default="8/64,4/64,4/32,4/16",
                   help="comma-separated bits/group")
    p.add_argument("--skip_embeddings", action="store_true",
                   help="mixed precision: leave embedding/output at fp32")
    p.add_argument("--max_new", type=int, default=112)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    torch.manual_seed(1337)
    temps = [float(t) for t in a.temperatures.split(",")]
    base = load_model(a.checkpoint, a.device)
    cats = categories()
    cat_id = cats.index(a.category)
    draw_b, jump_b = load_books()

    print(f"fp32 reference ({a.category})")
    draw_grid(base, cat_id, temps, a.n, draw_b, jump_b, a.device, a.max_new,
              f"q_{a.category}_fp32.png", f"{a.category} — fp32")

    for cfg in a.configs.split(","):
        bits, group = (int(x) for x in cfg.split("/"))
        torch.manual_seed(1337)          # same RNG stream as fp32
        m, nq, ns, err = quantize_model(base, bits, group, a.skip_embeddings)
        tag = f"int{bits}_g{group}" + ("_mixed" if a.skip_embeddings else "")
        print(f"\n{tag}: {nq} tensors quantized, {ns} skipped, "
              f"mean relative error {100*err:.2f}%")
        draw_grid(m, cat_id, temps, a.n, draw_b, jump_b, a.device, a.max_new,
                  f"q_{a.category}_{tag}.png",
                  f"{a.category} — {bits}-bit, group {group}  "
                  f"(err {100*err:.1f}%)")

    print("\nCompare against q_%s_fp32.png. The t=0.0 row is deterministic --"
          "\nif it matches fp32 exactly, quantization is lossless for that config."
          % a.category)


if __name__ == "__main__":
    main()
