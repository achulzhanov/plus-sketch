# plus-sketch

A 274K-parameter transformer that draws pictures, running on a **1986 Macintosh
Plus**: Motorola 68000 at 7.83 MHz, 1 MB of RAM, no floating-point unit, and a
512×342 one-bit display.

You type `draw a cat`. Three minutes later, there's a cat.

> Someone who saw the first samples said they looked like "a toddler trying to
> draw something." That is exactly right, and it is the point.

> **Status:** the model is trained, quantized, and verified. The 68000 port is
> in progress. See [Roadmap](#roadmap).

---

## Why this is possible at all

The obvious approaches don't fit. Estimated on a 68000 at ~5.9 MHz effective
(video DMA steals about a quarter of the bus):

| approach | time per image |
|---|---|
| diffusion (tiny U-Net, 32×32, 20 denoising steps) | ~3.2 hours |
| autoregressive pixels, 64×64 | ~1.1 hours |
| autoregressive pixels, full screen | ~48 hours |
| **vector strokes (this project)** | **~3 minutes** |

Diffusion dies because spatial dimensions multiply every layer's cost and you
pay it 20–50 times per image. Pixel generation dies because 512×342 is 175,104
tokens.

Vector strokes work because a drawing is ~55 *points*, not 175,000 pixels —
three orders of magnitude less work. And there is no resolution: the model
emits pen offsets, so the same sketch renders at any scale for free. It fills
the screen at no extra cost.

## How it works

Google's [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)
ships 50M human doodles in **stroke-3** format — each point is `(Δx, Δy,
pen_lift)`, a movement relative to the previous point.

k-means over the distribution of those movements produces a **codebook**: 384
representative offsets. Every delta becomes the index of its nearest entry. At
that moment "predict the next pen movement" is identical in form to "predict
the next word", and a standard decoder-only transformer handles it unmodified.

```
<BOS> <cat:bicycle> <off> <off> <PEN_UP> <off> ... <EOS>
```

The decoder on the Mac is a **1.5 KB lookup table**. Emit a token, index the
table, `LineTo` by that offset. No neural network on the output side — which is
exactly why this fits a 68000 and diffusion doesn't.

### Vocabulary (732 tokens)

| range | meaning |
|---|---|
| 0 | BOS |
| 1 | EOS (also padding) |
| 2 | PEN_UP |
| 3–258 | draw offsets (256 codes) |
| 259–386 | jump offsets (128 codes) |
| 387–731 | categories (345) |

**Two codebooks, not one.** Measured on `apple`: pen-down deltas have p50 24 /
p95 63 px; pen-up jumps have p50 43 / p95 226. A single k-means spends nearly
every centroid near the origin and lands jumps ~30 px off — which disconnects a
bicycle frame from its wheels. Splitting them fixes it.

A delta is a *jump* when the **previous** row had `pen == 1` ("lift after this
point"). Easy to get backwards.

## Model

```
dim=64  n_layers=5  n_heads=4  n_kv_heads=2  hidden=172
vocab=732  max_seq_len=112
274,112 parameters   (273,408 weights + 704 norm values)
```

Llama-style: RMSNorm, SwiGLU, RoPE, grouped-query attention, tied embedding and
output matrices. Architecture and training code are
[llama2.c](https://github.com/karpathy/llama2.c) unmodified — only the
tokenizer, data loader, and export are ours.

## Scripts

Everything runs from inside `training/`. llama2.c uses flat imports and
CWD-relative paths.

| script | what it does | when you run it |
|---|---|---|
| `sketchdata.py` | the data pipeline — four subcommands, below | building a dataset |
| `sketchtask.py` | dataset loader; replaces llama2.c's `tinystories.Task` | never directly, `train.py` imports it |
| `train.py` | training loop, from llama2.c (one line changed: the `Task` import) | once you have `.bin` shards |
| `sample.py` | draw from an fp32 checkpoint; renders PNG and SVG | evaluating a model, choosing temperature |
| `quantsweep.py` | fake-quantizes in memory and draws, across bit widths and group sizes | **before** writing any C |
| `export_fixed.py` | writes `.psk` files: int4, int8, or fp32 | once you're happy with the model |
| `verify_psk.py` | reads a `.psk` back, checks it, samples from it | after every export; also the spec for the C loader |

### `sketchdata.py` subcommands

- **`fetch`** — download Quick, Draw! sketch-rnn `.npz` files (~6 GB for all 345).
- **`codebook`** — k-means over pen deltas, writes `models/codebook.npz`. Runs
  once; the result is independent of length filters, so you never rebuild it
  when re-tokenizing.
- **`roundtrip`** — renders real drawings above their quantized versions.
  **This is the gate.** The bottom row *is* the training data and it is the
  model's ceiling; if it isn't recognizable, no amount of training will help.
- **`tokenize`** — `.npz` → `.bin` shards. Cheap and disposable; re-run freely
  to change `max_points` or `max_seq_len`.

### The three checkpoints

`out/ckpt.pt` is the fp32 PyTorch training checkpoint. `out/model.bin` is
llama2.c's flat fp32 dump, written at each eval. Neither runs on a Mac. The
`.psk` files are the deployable artifacts.

## Quantization

Weights are quantized to **4 bits, symmetric per-group scales, group size 64**,
scales stored as 16.16 fixed point. No floating-point arithmetic anywhere in
the inference path.

### int4 vs int8 on a Macintosh Plus

| | int8 + `MULS.W` | **int4 + 16-entry LUT** |
|---|---|---|
| cycles per MAC | ~100 | **~34** |
| seconds per token | 10.54 | **3.58** |
| file size | 288 KB | **155 KB** |
| apple / house (~45 tok) | 7.9 min | **2.7 min** |
| cat (~55 tok) | 9.7 min | **3.3 min** |
| bicycle (~75 tok) | 13.2 min | **4.5 min** |

**2.9× faster**, and the whole difference is one instruction. `MULS.W` costs
~70 of the ~100 cycles in an int8 multiply-accumulate. With 4-bit weights there
are only 16 possible values, so you precompute `activation × w` for all 16 once
per input column and the inner loop becomes unpack-nibble, index-table, add. No
multiply at all, and it's exact — no additional accuracy loss.

The 68000 has no scaled index addressing (that arrived with the 68020), so the
lookup needs `ADD.W D0,D0` before `MOVE.W (A2,D0.W),D2`.

Group size 64 rather than 32 or 16 for three reasons: scales cost 16 KB instead
of 33 or 65; the accumulator is rescaled once per group, and rescaling sits in
the hot loop that runs half a million times per token; and the visual difference
between group sizes is within sampling noise.

### Measured error

| config | mean | worst tensor |
|---|---|---|
| int8, group 64 | 0.66% | 0.78% (`layers.0.attention.wo`) |
| int4, group 64 | 12.05% | 14.05% (`layers.4.feed_forward.w2`) |

Norms are **never quantized** — 704 values, 2.8 KB at 16.16. Their measured
error is 7.63e-06, exactly the 16.16 rounding floor of 0.5/65536, which is a
tidy confirmation that the fixed-point conversion does what it claims.

**int8 is effectively lossless. int4 is a real ~12% perturbation, and the
drawings survive it** at the temperature we ship.

A trap worth naming: at temperature 0 the int4 output *looked better* than
fp32, and the config that looked best had the **highest** error of the three.
There is no mechanism by which a 12% weight perturbation improves a model.
Greedy decoding is a knife-edge — one shifted logit flips an argmax and the
whole trajectory cascades. Excellent as a determinism check, useless for
ranking configurations. Judge quantization at the temperature you'll ship.

### `.psk` file format

Self-contained: header, codebook, and weights in one file, so the 68k loader
needs no separate codebook and no SANE.

```
0    magic "PSK1", version, bits, group_size,
     dim, hidden_dim, n_layers, n_heads, n_kv_heads,
     vocab_size, max_seq_len, shared_classifier,
     n_draw, n_jump                        -> padded to 128 bytes
128  draw codebook   n_draw × 2 int16
     jump codebook   n_jump × 2 int16
     norms           16.16 int32, never quantized
     per tensor:  scales (int32 16.16), then values
```

Tensor order: `tok_embeddings`, then per layer `wq wk wv wo w1 w2 w3`, then
`output` if the classifier isn't tied.

**int4 packing:** two's-complement nibbles in [−8, 7]. Element 2i in the LOW
nibble of byte i, element 2i+1 in the HIGH nibble. Odd tail zero-padded. The C
unpacker must match exactly — `verify_psk.py:unpack_int4` is the reference.

## Macintosh Plus budget

Booting with the app replacing the Finder leaves roughly 850 KB:

| | |
|---|---|
| `plus_sketch_q4.psk` (weights + scales + norms + codebook) | 155 KB |
| KV cache | 35 KB |
| code, framebuffer, scratch | ~120 KB |
| **total** | **~310 KB** |

Memory was never the binding constraint. **int4 is a speed decision.**

## What we learned

**Temperature matters far more than in text.** Stroke coordinates are
*relative*, so every position is a running sum — one bad sample doesn't just
corrupt that token, it displaces everything after it. Text absorbs a strange
word; a drawing cannot absorb a misplaced pen.

The optimum sits around **0.15–0.3**, much narrower and much lower than the
0.7–1.0 typical for language models. It also varies by category:

| category | mean points | best temperature |
|---|---|---|
| house | 34 | 0.4 |
| apple | 38 | 0.3–0.4 |
| cat | 69 | 0.3 |
| bicycle | 68 | 0.15 |
| octopus | 72 | 0.2 |

Longer drawings generally want lower temperature — more tokens for error to
compound through — but length alone doesn't predict it. Cat and bicycle are the
same length and differ by a factor of two, because a cat is organic (a wobbly
ear is still an ear) while a bicycle is rigid (two similar wheels joined at
specific points). The build ships **0.2** as a default, with the intent to make
it a user-facing `NEAT / LOOSE / WILD` control — which doubles as a way to
teach anyone using it what sampling temperature does.

**Greedy decoding fails visibly.** At temperature 0 the model produces a shape
no human ever drew, identically every run. The mode of the distribution is not
a typical sample from it — and here you can *see* that, which you can't with
text.

**The training-set length filter selects, it doesn't truncate.** `max_points`
drops whole drawings above a threshold. Since `cat` and `bicycle` average ~69
points while `apple` and `house` average ~35, a 64-point filter kept 100% of
apples but only the rushed, incomplete 48% of bicycles. The model learned from
bad bicycles and drew two blobs. Raising to 96 fixed it. The filter is
simultaneously a length bound, a speed control, and a data-quality decision —
the third being the one that bites.

**Loss is not comparable across tokenization changes.** Padding tokens are
trivially predictable and drag the mean down, so a run with more padding shows
a lower loss at equal quality. Divide by the non-pad fraction before comparing,
and judge by the drawings regardless.

**The model learns patterns, not counts.** Octopus reliably produces a dome with
several parallel appendages beneath it — but 4–6 of them, not 8. A 5-layer
model has no obvious mechanism for "seven more times." Nobody notices.
Categories defined by *one closed contour plus a few marks* (face, tree, apple,
sun) work best; categories defined by *repeated countable parts* or *rigid
geometric relationships* (bicycle) are hardest.

**Training is launch-bound, not compute-bound.** At batch 128 an M-series GPU
sat at ~50% while achieving 2.3% of peak FLOPs and ~1% of memory bandwidth. A
5-layer transformer forward+backward is ~300 individual kernel dispatches; at
~100 µs each that's the entire iteration time. Raising batch size to 256 does
the same number of dispatches with more work each — but also does fewer
optimizer updates per token, which matters, so 256 rather than 512.

## Train your own

```bash
pip install torch numpy scikit-learn matplotlib

curl -L -o data/categories.txt \
  https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt

cd training
python sketchdata.py fetch                          # ~6 GB, 345 categories
python sketchdata.py codebook --sample=20000        # k-means, runs once
python sketchdata.py roundtrip --category=bicycle   # the gate
python sketchdata.py tokenize --vocab_size=732 --max_points=96 \
                              --max_seq_len=112 --per_category=5000

python train.py --out_dir=out --vocab_source=custom --vocab_size=732 \
  --dim=64 --n_layers=5 --n_heads=4 --n_kv_heads=2 --multiple_of=4 \
  --max_seq_len=112 --batch_size=256 --gradient_accumulation_steps=1 \
  --learning_rate=1.5e-3 --dropout=0.05 --weight_decay=0.01 --beta2=0.99 \
  --warmup_iters=1000 --max_iters=123819 \
  --eval_interval=2000 --eval_iters=100 \
  --device=mps --dtype=float32 --compile=False

python sample.py --checkpoint=out/ckpt.pt --category=cat --n=5 --png --temperature=0.2
python quantsweep.py --checkpoint=out/ckpt.pt --category=bicycle
python export_fixed.py --checkpoint=out/ckpt.pt --bits=4
python verify_psk.py --psk=../models/plus_sketch_q4.psk --draw=cat
```

1.58M drawings, 177M tokens, 20 epochs ≈ 13,000 tokens per parameter, 1.5–2
hours on an M-series laptop.

### Design notes

- Drawings are padded to exactly `max_seq_len`, because llama2.c's dataset takes
  *aligned* windows. One window = one complete drawing, never a slice across
  two. Costs ~50% of tokens to padding; worth it.
- `sample.py` masks category tokens out of the logits after position 1, or the
  model starts drawing a house halfway through a cat.
- Typing, never a menu. A menu implies stored clip art being traced; typing
  implies generation. Unknown words get `I DON'T KNOW HOW TO DRAW A ___.` —
  the honest decline is a feature.
- Precompute `1/T` as 16.16 rather than dividing 732 logits per token. `DIVU`
  is ~140 cycles on a 68000.

## Hardware compatibility

Built for the **68000**, so it runs on everything upward: Plus, SE, Classic,
Portable, and all later 68k Macs. The reverse isn't true — a 68020+ binary
won't run on a Plus.

The 68020 added 32-bit addressing (the 68000 only wires 24 lines, capping it at
16 MB), scaled index addressing, an instruction cache, and a much faster
multiply (~28 cycles vs ~70). The 68030 adds an MMU and data cache. A **Mac
SE/30** should manage roughly **1.4 s/token — a cat in about 78 seconds**
against the Plus's 3.3 minutes.

## Roadmap

- [x] Data pipeline: fetch, codebook, round-trip validation, tokenize
- [x] Training and macOS sampling
- [x] Quantization validated in PyTorch (int4, group 64)
- [x] `.psk` export and verification (int4, int8, fp32)
- [ ] `core/` — pure-integer C, no float anywhere, verified against `sample.py`
- [ ] Retro68 cross-compile, Mini vMac
- [ ] QuickDraw rendering, keyboard input
- [ ] Real hardware

## Prior art

The tokenization here is essentially **Sketchformer** (Ribeiro, Bui, Collomosse
& Ponti, CVPR 2020), which built a k-means dictionary over `(Δx, Δy)` with
K=1000 and deliberately oversampled inter-stroke transitions — the same problem
our two-codebook split solves differently. **SketchGPT** (Tiwari, Biswas &
Lladós, ICDAR 2024) does autoregressive sketch generation. The original
**sketch-rnn** (Ha & Eck, 2017) used an LSTM VAE with continuous outputs.

None of the machine learning here is novel. Running it on a 68000 with 1 MB and
no FPU is.

## Attribution

`model.py`, `train.py`, `configurator.py`, and `export.py` derive from
[llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy (MIT).
Training data is Google's Quick, Draw!. See `NOTICE`.

Google's dataset is called Quick, Draw!. Apple's 2D graphics library, in the
Macintosh Plus ROM since 1986, is called QuickDraw. This project uses the
second to render the first.
