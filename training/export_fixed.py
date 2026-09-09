#!/usr/bin/env python3
"""
export_fixed.py -- plus-sketch model files for the 68000 port.

Emits one self-contained .psk file: header, codebook, and weights. Nothing in
it is floating point, so core/ never needs SANE.

    python export_fixed.py --checkpoint=out/ckpt.pt --bits=4
    python export_fixed.py --checkpoint=out/ckpt.pt --bits=8
    python export_fixed.py --checkpoint=out/ckpt.pt --bits=32   # reference

FILE FORMAT (all little-endian, which is wrong-endian for 68k -- the loader
byte-swaps, see docs. Chosen so the file is trivially inspectable on the host.)

  offset 0    magic  "PSK1"                          4 bytes
              version                                int32 = 1
              bits (4 | 8 | 32)                      int32
              group_size                             int32
              dim, hidden_dim, n_layers,
              n_heads, n_kv_heads, vocab_size,
              max_seq_len                            7 x int32
              shared_classifier (0|1)                int32
              n_draw, n_jump                         2 x int32
              reserved                               pad to 128 bytes

  then        draw codebook   n_draw x 2 int16   (dx, dy)
              jump codebook   n_jump x 2 int16

  then        norms, 16.16 fixed, NEVER quantized (2*dim per layer + dim,
              tiny and highly sensitive)

  then        for each quantized tensor, in this order:
                token_embeddings, then per layer wq wk wv wo w1 w2 w3,
                then output (only if not shared_classifier)
              each stored as:
                scales    ceil(numel/group) x int32   (16.16)
                values    int8, or int4 packed two per byte

INT4 PACKING: two's-complement nibbles in [-8, 7]. Element 2i goes in the LOW
nibble of byte i, element 2i+1 in the HIGH nibble. Odd tail is zero-padded.
The C unpacker must match this exactly.
"""

import argparse
import os
import struct

import numpy as np
import torch

MAGIC = b"PSK1"
VERSION = 1
HEADER_BYTES = 128
FX_ONE = 65536.0          # 16.16


def to_fx(x):
    """float -> 16.16 int32, saturating."""
    v = np.round(np.asarray(x, dtype=np.float64) * FX_ONE)
    return np.clip(v, -2**31, 2**31 - 1).astype(np.int32)


def quantize(w, bits, group):
    """
    Symmetric per-group. Returns (q int32 in range, scales float64).
    Matches quantsweep.fake_quant so sweep results carry over exactly.
    """
    flat = w.reshape(-1).astype(np.float64)
    pad = (-len(flat)) % group
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    g = flat.reshape(-1, group)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(g).max(axis=1) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.round(g / scale[:, None]), -qmax - 1, qmax).astype(np.int32)
    return q.reshape(-1)[: len(flat) - pad if pad else len(flat)], scale


def pack_int4(q):
    """int32 values in [-8,7] -> two's-complement nibbles, low nibble first."""
    q = np.asarray(q, dtype=np.int32)
    if len(q) % 2:
        q = np.concatenate([q, [0]])
    lo = (q[0::2] & 0xF).astype(np.uint8)
    hi = (q[1::2] & 0xF).astype(np.uint8)
    return (lo | (hi << 4)).tobytes()


def tensor_order(sd, n_layers, shared):
    """The exact order core/ will read them back in."""
    out = [("tok_embeddings", sd["tok_embeddings.weight"])]
    for i in range(n_layers):
        p = f"layers.{i}."
        for n in ("attention.wq", "attention.wk", "attention.wv", "attention.wo",
                  "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"):
            out.append((f"{p}{n}", sd[p + n + ".weight"]))
    if not shared:
        out.append(("output", sd["output.weight"]))
    return out


def norm_order(sd, n_layers):
    out = []
    for i in range(n_layers):
        out.append((f"layers.{i}.attention_norm", sd[f"layers.{i}.attention_norm.weight"]))
        out.append((f"layers.{i}.ffn_norm", sd[f"layers.{i}.ffn_norm.weight"]))
    out.append(("norm", sd["norm.weight"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/ckpt.pt")
    ap.add_argument("--codebook", default="../models/codebook.npz")
    ap.add_argument("--out", default="")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 32])
    ap.add_argument("--group", type=int, default=64)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["model_args"]
    sd = ck["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    sd = {k: v.float().numpy() for k, v in sd.items()}

    dim = cfg["dim"]
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    n_kv_heads = cfg.get("n_kv_heads") or n_heads
    vocab = cfg["vocab_size"]
    seq = cfg["max_seq_len"]
    hidden = sd["layers.0.feed_forward.w1.weight"].shape[0]
    shared = "output.weight" not in sd or np.shares_memory(
        sd["tok_embeddings.weight"], sd.get("output.weight", np.array([])))
    if "output.weight" in sd:
        shared = np.array_equal(sd["output.weight"], sd["tok_embeddings.weight"])

    cb = np.load(a.codebook)
    draw_b, jump_b = cb["draw"].astype(np.int16), cb["jump"].astype(np.int16)

    out_path = a.out or f"../models/plus_sketch_q{a.bits}.psk" if a.bits != 32 \
        else a.out or "../models/plus_sketch_fp32.psk"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"dim={dim} hidden={hidden} layers={n_layers} heads={n_heads} "
          f"kv_heads={n_kv_heads} vocab={vocab} seq={seq} shared={shared}")

    with open(out_path, "wb") as f:
        head = struct.pack("<4siiiiiiiiiiiii", MAGIC, VERSION, a.bits,
                           a.group if a.bits != 32 else 0,
                           dim, hidden, n_layers, n_heads, n_kv_heads,
                           vocab, seq, int(shared), len(draw_b), len(jump_b))
        f.write(head + b"\0" * (HEADER_BYTES - len(head)))

        f.write(draw_b.astype("<i2").tobytes())
        f.write(jump_b.astype("<i2").tobytes())

        # norms: always 16.16, never quantized
        n_norm = 0
        for name, w in norm_order(sd, n_layers):
            f.write(to_fx(w).astype("<i4").tobytes())
            n_norm += w.size

        # weights
        total_q = 0
        smin, smax = np.inf, 0.0
        errs = []
        for name, w in tensor_order(sd, n_layers, shared):
            if a.bits == 32:
                f.write(w.astype("<f4").tobytes())
                total_q += w.size
                continue
            q, scale = quantize(w, a.bits, a.group)
            smin, smax = min(smin, scale.min()), max(smax, scale.max())
            deq = (q.reshape(-1)[: w.size] *
                   np.repeat(scale, a.group)[: w.size]).reshape(w.shape)
            errs.append(np.abs(deq - w).mean() / np.abs(w).mean())
            f.write(to_fx(scale).astype("<i4").tobytes())
            if a.bits == 8:
                f.write(np.clip(q, -128, 127).astype(np.int8).tobytes())
            else:
                f.write(pack_int4(q))
            total_q += w.size

    size = os.path.getsize(out_path)
    print(f"\nwrote {out_path}  {size/1024:.1f} KB")
    print(f"  {total_q:,} weights + {n_norm:,} norm values")
    print(f"  codebook: {len(draw_b)} draw + {len(jump_b)} jump = "
          f"{2*(len(draw_b)+len(jump_b))*2/1024:.1f} KB")
    if a.bits != 32:
        print(f"  mean relative error: {100*np.mean(errs):.2f}%")
        print(f"  scale range: {smin:.3e} .. {smax:.3e}  "
              f"(16.16 -> {smin*FX_ONE:.1f} .. {smax*FX_ONE:.1f})")
        if smin * FX_ONE < 16:
            print("  WARNING: smallest scale is near the 16.16 floor -- "
                  "consider a different fixed-point format")


if __name__ == "__main__":
    main()
