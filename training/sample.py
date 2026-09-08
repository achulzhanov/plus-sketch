#!/usr/bin/env python3
"""
sample.py -- draw from a trained plus-sketch checkpoint.

The macOS reference renderer. Whatever this produces is what core/sketch.c
plus QuickDraw MoveTo/LineTo must reproduce on the Plus, so the SVG output is
the golden artifact to diff the 68k port against.

    python sample.py --checkpoint=out/ckpt.pt --category=cat
    python sample.py --checkpoint=out/ckpt.pt --category=house --n=4 --svg
    python sample.py --checkpoint=out/ckpt.pt            # interactive
"""

import argparse
import difflib
import os

import numpy as np
import torch

from model import ModelArgs, Transformer
from sketchdata import (BOS, EOS, PEN_UP, CAT_BASE, DRAW_BASE, JUMP_BASE,
                        categories, load_books, decode)

# Hand-written aliases. The 68k build needs the same table -- category names
# are sometimes awkward ("hot air balloon", "The Mona Lisa").
ALIASES = {
    "kitty": "cat", "kitten": "cat", "puppy": "dog", "auto": "car",
    "automobile": "car", "bike": "bicycle", "home": "house",
    "flower": "flower", "person": "face", "boat": "sailboat",
}


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = Transformer(ModelArgs(**ck["model_args"]))
    sd = ck["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    m.load_state_dict(sd, strict=False)
    return m.eval().to(device)


def resolve(word, cats):
    """exact -> alias -> plural strip -> fuzzy -> None"""
    w = word.strip().lower()
    if w in cats:
        return w
    if w in ALIASES and ALIASES[w] in cats:
        return ALIASES[w]
    if w.endswith("s") and w[:-1] in cats:
        return w[:-1]
    near = difflib.get_close_matches(w, cats, n=1, cutoff=0.8)
    return near[0] if near else None


@torch.no_grad()
def generate(model, cat_id, temperature, max_new, device):
    toks = [BOS, CAT_BASE + cat_id]
    seq = model.params.max_seq_len
    for _ in range(max_new):
        idx = torch.tensor([toks[-seq:]], dtype=torch.long, device=device)
        logits = model(idx)[:, -1, :].float().squeeze(0)
        logits[BOS] = -float("inf")
        logits[CAT_BASE:] = -float("inf")      # never emit a category mid-draw
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            nxt = int(torch.multinomial(
                torch.softmax(logits / temperature, dim=-1), 1))
        if nxt == EOS:
            break
        toks.append(nxt)
    return toks


def normalize(strokes, size=512, margin=24):
    pts = [p for s in strokes for p in s]
    if not pts:
        return strokes
    xs, ys = zip(*pts)
    w, h = max(max(xs) - min(xs), 1), max(max(ys) - min(ys), 1)
    k = (size - 2 * margin) / max(w, h)
    ox, oy = min(xs), min(ys)
    return [[(margin + (x - ox) * k, margin + (y - oy) * k) for x, y in s]
            for s in strokes]


def write_svg(strokes, path, size=512):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
             f'height="{size}" viewBox="0 0 {size} {size}">',
             f'<rect width="{size}" height="{size}" fill="white"/>']
    for s in strokes:
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                     for i, (x, y) in enumerate(s))
        parts.append(f'<path d="{d}" stroke="black" stroke-width="3" '
                     f'fill="none" stroke-linecap="round"/>')
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))


def draw_one(model, cats, word, a, draw_b, jump_b, tag=""):
    hit = resolve(word, cats)
    if hit is None:
        print(f"  I DON'T KNOW HOW TO DRAW A {word.upper()}.")
        return None
    if hit != word.strip().lower():
        print(f"  ({word} -> {hit})")
    toks = generate(model, cats.index(hit), a.temperature, a.max_new, a.device)
    strokes = normalize(decode(toks, draw_b, jump_b))
    npts = sum(len(s) for s in strokes)
    print(f"  {hit}: {len(toks)} tokens, {len(strokes)} strokes, {npts} points"
          f"  (~{len(toks) * 3.4 / 60:.1f} min on the Plus)")
    base = f"{hit.replace(' ', '_')}_t{a.temperature}{tag}"
    if a.svg:
        write_svg(strokes, base + ".svg")
        print(f"  wrote {base}.svg")
    return strokes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="out/ckpt.pt")
    p.add_argument("--category", default="")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new", type=int, default=80)
    p.add_argument("--device", default="cpu")
    p.add_argument("--svg", action="store_true")
    p.add_argument("--png", action="store_true")
    a = p.parse_args()

    model = load_model(a.checkpoint, a.device)
    cats = categories()
    draw_b, jump_b = load_books()
    print(f"loaded {a.checkpoint}: {len(cats)} categories\n")

    def render_png(all_strokes, labels, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(all_strokes)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
        for ax, strokes, lab in zip(axes[0], all_strokes, labels):
            for s in strokes:
                xs, ys = zip(*s)
                ax.plot(xs, ys, "k-", lw=2.5)
            ax.invert_yaxis(); ax.axis("equal"); ax.axis("off")
            ax.set_title(lab, fontsize=9)
        plt.savefig(path, dpi=80, bbox_inches="tight")
        print(f"wrote {path}")

    if a.category:
        got, labels = [], []
        for i in range(a.n):
            s = draw_one(model, cats, a.category, a, draw_b, jump_b,
                         tag=f"_{i}" if a.n > 1 else "")
            if s:
                got.append(s); labels.append(f"{a.category} #{i+1}")
        if a.png and got:
            render_png(got, labels, f"{a.category.replace(' ', '_')}_t{a.temperature}.png")
        return

    print("DRAW A ___   (blank line to quit)")
    while True:
        try:
            w = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not w:
            break
        s = draw_one(model, cats, w, a, draw_b, jump_b)
        if a.png and s:
            render_png(got, labels, f"{a.category.replace(' ', '_')}_t{a.temperature}.png")


if __name__ == "__main__":
    main()
