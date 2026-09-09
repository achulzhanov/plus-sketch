#!/usr/bin/env python3
"""
dump_ref.py -- reference logits for the C port to check against.

Loads the SAME quantized weights the C reads (straight out of the .psk) and
runs them through PyTorch. With --quant_acts it also quantizes activations to
int8 per group exactly the way core/forward.c:quantize_vec does, so the only
remaining difference is 16.16 fixed-point arithmetic. Without it, the
reference keeps activations in float and the C will look worse than it is.

    python dump_ref.py --psk=../models/plus_sketch_q4.psk \\
        --out=../tests/ref_q4.bin --quant_acts

Format (little-endian, same convention as .psk):
    "PREF"   4 bytes
    n_steps  int32
    vocab    int32
    tokens   n_steps x int32
    logits   n_steps x vocab x float32     (logits AFTER feeding tokens[i])
"""

import argparse
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

from model import ModelArgs, Transformer
from verify_psk import read_psk

FX_ONE = 65536.0


def quant_act(x, group):
    """
    Mirror core/forward.c:quantize_vec.

      scale = ceil(max|x| * 65536 / 127) / 65536   -- 16.16, rounded up so
                                                     |q| can't exceed 127
      q     = round(x / scale), clamped to [-127, 127]

    Padding to a group multiple matches the C, which zero-fills the tail.
    """
    n = x.shape[-1]
    gpr = -(-n // group)
    pad = gpr * group - n
    xp = F.pad(x, (0, pad)) if pad else x
    g = xp.reshape(*x.shape[:-1], gpr, group)

    s = torch.ceil(g.abs().amax(-1, keepdim=True) * (FX_ONE * 256) / 127) / (FX_ONE * 256)
    s = torch.where(s <= 0, torch.ones_like(s), s)
    q = torch.clamp(torch.round(g / s), -127, 127)

    out = (q * s).reshape(*x.shape[:-1], gpr * group)
    return out[..., :n]


class QuantLinear(torch.nn.Module):
    """Linear that quantizes its input the way the C does."""

    def __init__(self, lin, group):
        super().__init__()
        self.weight = lin.weight
        self.group = group

    def forward(self, x):
        return F.linear(quant_act(x, self.group), self.weight)


def wrap_quant_acts(m, group):
    n = 0
    for layer in m.layers:
        for mod, name in ((layer.attention, "wq"), (layer.attention, "wk"),
                          (layer.attention, "wv"), (layer.attention, "wo"),
                          (layer.feed_forward, "w1"),
                          (layer.feed_forward, "w2"),
                          (layer.feed_forward, "w3")):
            setattr(mod, name, QuantLinear(getattr(mod, name), group))
            n += 1
    m.output = QuantLinear(m.output, group)
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psk", default="../models/plus_sketch_q4.psk")
    ap.add_argument("--out", default="../tests/ref_q4.bin")
    ap.add_argument("--category", type=int, default=3,
                    help="category index, not token id")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--quant_acts", action="store_true",
                    help="quantize activations to int8 like the C does")
    a = ap.parse_args()

    cfg, sd, draw_b, jump_b = read_psk(a.psk)

    args = ModelArgs(dim=cfg["dim"], n_layers=cfg["n_layers"],
                     n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"],
                     vocab_size=cfg["vocab_size"], hidden_dim=cfg["hidden_dim"],
                     max_seq_len=cfg["max_seq_len"], dropout=0.0)
    m = Transformer(args)
    state = {k: torch.tensor(v, dtype=torch.float32) for k, v in sd.items()}
    if cfg["shared"]:
        state["output.weight"] = state["tok_embeddings.weight"]
    _, unexpected = m.load_state_dict(state, strict=False)
    if unexpected:
        print("unexpected keys:", unexpected)
    m.eval()

    if a.quant_acts:
        n = wrap_quant_acts(m, cfg["group"])
        print(f"\nactivation quantization ON: wrapped {n} linear layers "
              f"(group {cfg['group']})")
    else:
        print("\nactivation quantization OFF -- the C will look worse than it "
              "is.\n  Rerun with --quant_acts for a fair comparison.")

    # Fixed, arbitrary but valid sequence. Exercises draw tokens, a jump,
    # and PEN_UP. Deterministic on both sides.
    draw_base = 3
    jump_base = draw_base + len(draw_b)
    cat_base = jump_base + len(jump_b)
    toks = [0, cat_base + a.category,
            draw_base + 10, draw_base + 200, draw_base + 55,
            2,                                   # PEN_UP
            jump_base + 7,
            draw_base + 3, draw_base + 128, draw_base + 91,
            2, draw_base + 40][:a.steps]

    print(f"\n{len(toks)} steps: {toks}")

    rows = []
    with torch.no_grad():
        for i in range(len(toks)):
            idx = torch.tensor([toks[:i + 1]], dtype=torch.long)
            lg = m(idx)[0, -1, :].float().numpy()
            rows.append(lg)
            order = np.argsort(-lg)[:3]
            gap = float(lg[order[0]] - lg[order[1]])
            print(f"  pos {i:2d} tok {toks[i]:4d} -> top3 "
                  f"{[(int(t), round(float(lg[t]), 3)) for t in order]}"
                  f"   gap {gap:.3f}"
                  f"{'  <- near-tie' if gap < 0.05 else ''}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(b"PREF")
        f.write(struct.pack("<ii", len(toks), cfg["vocab_size"]))
        f.write(np.array(toks, dtype="<i4").tobytes())
        f.write(np.array(rows, dtype="<f4").tobytes())
    print(f"\nwrote {a.out}  ({os.path.getsize(a.out)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
