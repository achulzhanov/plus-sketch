#!/usr/bin/env python3
"""
sketchdata.py -- Quick, Draw! -> tokens for plus-sketch.

TOKEN LAYOUT (vocab 604)
    0                BOS
    1                EOS   (also used as padding)
    2                PEN_UP
    3   .. 194       draw offsets   (192 codes)
    195 .. 258       jump offsets   ( 64 codes)
    259 .. 603       categories     (345)

Two codebooks because the distributions are genuinely different -- measured on
apple: pen-down moves have p50 24 / p95 63, pen-up jumps p50 43 / p95 226. One
k-means over both would spend nearly every centroid near the origin and place
jumps tens of pixels wrong, which lands whole strokes in the wrong spot.

A delta is a JUMP when the PREVIOUS row had pen==1 ("lift after this point").

    python sketchdata.py fetch --categories=cat,house,fish
    python sketchdata.py codebook
    python sketchdata.py roundtrip --category=apple      # <- the gate
    python sketchdata.py tokenize
"""

import argparse
import glob
import json
import os
import urllib.request

import numpy as np

DATA = "../data"
NPZ = os.path.join(DATA, "npz")
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn"

BOS, EOS, PEN_UP = 0, 1, 2
N_DRAW, N_JUMP = 256, 128
DRAW_BASE = 3
JUMP_BASE = DRAW_BASE + N_DRAW
CAT_BASE = JUMP_BASE + N_JUMP


# -------------------------------------------------------------------------
def categories():
    with open(os.path.join(DATA, "categories.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def load_npz(cat, split="train"):
    p = os.path.join(NPZ, f"{cat}.npz")
    return np.load(p, encoding="latin1", allow_pickle=True)[split]


def split_deltas(s):
    """-> [(dx, dy, is_jump, lifts_after), ...] for one drawing."""
    out, prev_lift = [], False
    for dx, dy, pen in s:
        out.append((int(dx), int(dy), prev_lift, pen == 1))
        prev_lift = pen == 1
    return out


# -------------------------------------------------------------------------
def cmd_fetch(a):
    os.makedirs(NPZ, exist_ok=True)
    cats = [c.strip() for c in a.categories.split(",")] if a.categories else categories()
    for i, c in enumerate(cats, 1):
        dst = os.path.join(NPZ, f"{c}.npz")
        if os.path.exists(dst):
            continue
        url = f"{BASE_URL}/{c.replace(' ', '%20')}.npz"
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"[{i}/{len(cats)}] {c}  {os.path.getsize(dst)/1e6:.1f} MB")
        except Exception as e:
            print(f"[{i}/{len(cats)}] {c}  FAILED: {e}")
    print(f"\n{len(glob.glob(os.path.join(NPZ, '*.npz')))} files in {NPZ}")


# -------------------------------------------------------------------------
def cmd_codebook(a):
    from sklearn.cluster import MiniBatchKMeans

    files = sorted(glob.glob(os.path.join(NPZ, "*.npz")))
    assert files, f"no .npz in {NPZ} -- run fetch first"
    draws, jumps = [], []
    per_cat = a.sample // len(files) + 1
    for f in files:
        for s in load_npz(os.path.basename(f)[:-4])[:per_cat]:
            for dx, dy, is_jump, _ in split_deltas(s):
                (jumps if is_jump else draws).append((dx, dy))
    draws, jumps = np.array(draws, float), np.array(jumps, float)
    print(f"{len(draws):,} pen-down deltas, {len(jumps):,} jumps "
          f"from {len(files)} categories")

    books = {}
    for name, pts, k in (("draw", draws, N_DRAW), ("jump", jumps, N_JUMP)):
        # clip the extreme tail so centroids pack where the mass is
        lim = np.percentile(np.abs(pts), a.clip_pct)
        clipped = np.clip(pts, -lim, lim)
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10,
                             batch_size=4096).fit(clipped)
        c = np.round(km.cluster_centers_).astype(np.int16)
        books[name] = c
        err = np.abs(clipped - c[km.predict(clipped)]).max(axis=1)
        print(f"  {name}: {k} codes, clip +/-{lim:.0f}, "
              f"quant error p50 {np.percentile(err,50):.1f} "
              f"p95 {np.percentile(err,95):.1f} max {err.max():.1f} px")

    np.savez(os.path.join("../models", "codebook.npz"),
             draw=books["draw"], jump=books["jump"])
    print("\nwrote ../models/codebook.npz")


def load_books():
    d = np.load(os.path.join("../models", "codebook.npz"))
    return d["draw"].astype(np.int32), d["jump"].astype(np.int32)


def nearest(book, dx, dy):
    return int(np.argmin((book[:, 0] - dx) ** 2 + (book[:, 1] - dy) ** 2))


# -------------------------------------------------------------------------
def encode(s, cat_id, draw_b, jump_b):
    toks = [BOS, CAT_BASE + cat_id]
    for dx, dy, is_jump, lifts in split_deltas(s):
        if is_jump:
            toks.append(JUMP_BASE + nearest(jump_b, dx, dy))
        else:
            toks.append(DRAW_BASE + nearest(draw_b, dx, dy))
        if lifts:
            toks.append(PEN_UP)
    toks.append(EOS)
    return toks

def encode_fast(s, cat_id, draw_b, jump_b):
    d = s[:, :2].astype(np.int32)
    pen = s[:, 2].astype(np.int32)
    is_jump = np.zeros(len(s), bool)
    is_jump[1:] = pen[:-1] == 1
    idx = np.empty(len(s), np.int32)
    for mask, book, base in ((~is_jump, draw_b, DRAW_BASE), (is_jump, jump_b, JUMP_BASE)):
        if mask.any():
            dd = d[mask]
            idx[mask] = base + np.argmin(((dd[:, None, :] - book[None]) ** 2).sum(-1), 1)
    toks = [BOS, CAT_BASE + cat_id]
    for i in range(len(s)):
        toks.append(int(idx[i]))
        if pen[i] == 1:
            toks.append(PEN_UP)
    return toks + [EOS]

def decode(toks, draw_b, jump_b):
    """-> list of polylines [[(x,y), ...], ...]"""
    strokes, cur = [], [(0, 0)]
    x = y = 0
    for t in toks:
        if t in (BOS, EOS) or t >= CAT_BASE:
            continue
        if t == PEN_UP:
            if len(cur) > 1:
                strokes.append(cur)
            cur = [(x, y)]
            continue
        book, base = (jump_b, JUMP_BASE) if t >= JUMP_BASE else (draw_b, DRAW_BASE)
        dx, dy = book[t - base]
        x, y = x + int(dx), y + int(dy)
        if base == JUMP_BASE:
            cur = [(x, y)]          # jump starts a new stroke
        else:
            cur.append((x, y))
    if len(cur) > 1:
        strokes.append(cur)
    return strokes


def raw_strokes(s):
    strokes, cur = [], [(0, 0)]
    x = y = 0
    for dx, dy, pen in s:
        x, y = x + int(dx), y + int(dy)
        cur.append((x, y))
        if pen == 1:
            strokes.append(cur)
            cur = [(x, y)]
    if len(cur) > 1:
        strokes.append(cur)
    return strokes


def cmd_roundtrip(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    draw_b, jump_b = load_books()
    data = load_npz(a.category)
    n = a.n
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for j in range(n):
        s = data[j]
        for row, strokes, title in ((0, raw_strokes(s), "original"),
                                    (1, decode(encode(s, 0, draw_b, jump_b),
                                               draw_b, jump_b), "quantized")):
            ax = axes[row][j]
            for st in strokes:
                xs, ys = zip(*st)
                ax.plot(xs, ys, "k-", lw=2)
            ax.invert_yaxis(); ax.axis("equal"); ax.axis("off")
            if j == 0:
                ax.set_title(title, loc="left")
    out = f"roundtrip_{a.category}.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    print(f"wrote {out} -- if the bottom row is wrong, raise N_DRAW/N_JUMP")


# -------------------------------------------------------------------------
def cmd_tokenize(a):
    draw_b, jump_b = load_books()
    cats = categories()
    cat_idx = {c: i for i, c in enumerate(cats)}
    out_dir = os.path.join(DATA, f"tok{a.vocab_size}")
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*.bin")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(NPZ, "*.npz")))
    seqs, kept, dropped = [], 0, 0
    for fi, f in enumerate(files, 1):
        cat = os.path.basename(f)[:-4]
        if cat not in cat_idx:
            print(f"  skip {cat}: not in categories.txt")
            continue
        for s in load_npz(cat)[:a.per_category]:
            if len(s) > a.max_points:
                dropped += 1
                continue
            t = encode_fast(s, cat_idx[cat], draw_b, jump_b)
            if len(t) > a.max_seq_len:
                dropped += 1
                continue
            t += [EOS] * (a.max_seq_len - len(t))   # pad so aligned windows
            seqs.append(t)                           # = exactly one drawing
            kept += 1
        print(f"[{fi}/{len(files)}] {cat}: {kept:,} kept", flush=True)

    rng = np.random.default_rng(1337)
    rng.shuffle(seqs)
    arr = np.array(seqs, dtype=np.uint16)
    per = max(1, len(arr) // a.shards)
    for i in range(a.shards):
        chunk = arr[i * per: (i + 1) * per if i < a.shards - 1 else len(arr)]
        chunk.reshape(-1).tofile(os.path.join(out_dir, f"data{i:02d}.bin"))

    real = (arr != EOS).sum(axis=1)
    print(f"\n{kept:,} drawings kept, {dropped:,} dropped (>{a.max_points} pts)")
    print(f"tokens: {arr.size:,} total, {real.mean():.0f} real per drawing "
          f"({100*real.sum()/arr.size:.0f}% non-pad)")
    print(f"-> {out_dir} ({a.shards} shards, shard00 = val)")
    tpi = 128 * a.max_seq_len
    print(f"\nAt batch 128 x seq {a.max_seq_len} = {tpi:,} tok/iter: "
          f"1 epoch = {arr.size/tpi:,.0f} iters")
    for ep in (5, 10, 20):
        print(f"  {ep:2d} epochs -> --max_iters={int(arr.size/tpi*ep):,}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--categories", default="")
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("codebook")
    c.add_argument("--sample", type=int, default=200000)
    c.add_argument("--clip_pct", type=float, default=99.5)
    c.set_defaults(func=cmd_codebook)

    r = sub.add_parser("roundtrip")
    r.add_argument("--category", default="apple")
    r.add_argument("--n", type=int, default=5)
    r.set_defaults(func=cmd_roundtrip)

    t = sub.add_parser("tokenize")
    t.add_argument("--max_points", type=int, default=64)
    t.add_argument("--max_seq_len", type=int, default=80)
    t.add_argument("--per_category", type=int, default=70000)
    t.add_argument("--vocab_size", type=int, default=604)
    t.add_argument("--shards", type=int, default=10)
    t.set_defaults(func=cmd_tokenize)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
