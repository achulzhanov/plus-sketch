# plus-sketch

A 274K-parameter transformer that draws pictures, running on a **1986 Macintosh
Plus**: Motorola 68000 at 7.83 MHz, 1 MB of RAM, no floating-point unit, and a
512×342 one-bit display.

You type `draw a cat`. Two minutes later, there's a cat.

**Measured: 1.89 s/token.** A house takes about 45 seconds, a cat a little over
two minutes. No floating-point arithmetic anywhere in the inference path.

---

## Run it

`dist/PlusSketch.dsk` is an 800K floppy image with the application, the
quantized model, and the category list on it.

1. Get [Mini vMac](https://github.com/minivmac/minivmac) built for a Macintosh
   Plus with 1 MB, plus a Plus ROM (`vMac.ROM`) and a System 6.0.8 boot image.
2. Boot System 6 first, then **Control-O** `PlusSketch.dsk` to mount it
   alongside. Mounting it first makes the Mac try to boot from it and fail.
3. Run `PlusSketch`. Type a word, Return to draw. Left and right arrows change
   temperature; backspace edits; escape clears.

The disk is not bootable and has no SCSI driver partition, so it is for
emulators and floppy transfer — **not BlueSCSI**. For real hardware, copy the
three files onto an image made by Disk Jockey or a premade BlueSCSI image.

## Why this is possible at all

The obvious approaches don't fit. Estimated on a 68000 at ~5.9 MHz effective
(video DMA steals about a quarter of the bus):

| approach | time per image |
|---|---|
| diffusion (tiny U-Net, 32×32, 20 denoising steps) | ~3.2 hours |
| autoregressive pixels, 64×64 | ~1.1 hours |
| autoregressive pixels, full screen | ~48 hours |
| **vector strokes (this project)** | **~2 minutes** |

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
output matrices. Only `model.py`, `train.py`, `configurator.py` and `export.py`
come from [llama2.c](https://github.com/karpathy/llama2.c). The tokenizer, the
data loader, the export format, and the whole of `core/` are new.

## Performance

Measured on Mini vMac emulating a Macintosh Plus at authentic 1× speed.

| | s/token | a house (22 tok) | a cat (60 tok) |
|---|---|---|---|
| first working build | 30.11 | 11 min | 30 min |
| **after optimization** | **1.89** | **45 s** | **1.9 min** |

**16× faster**, and almost all of it came from one line. See
[porting notes](#porting-notes).

### Binary size

| | |
|---|---|
| first build (RetroConsole + float `printf`) | 1,259,385 bytes |
| **current** | **24,373 bytes** |

A single `%f` in `printf` pulls in newlib's floating-point formatting, which on
a machine with no FPU means SANE. That alone was most of the 1.2 MB.

## Quantization

Weights are quantized to **4 bits, symmetric per-group scales, group size 64**.
Scales are fixed point: 16.16 for weights, 8.24 for activations. No
floating-point arithmetic anywhere in the inference path.

### Measured error

| config | mean | worst tensor |
|---|---|---|
| int8, group 64 | 0.66% | 0.78% (`layers.0.attention.wo`) |
| int4, group 64 | 12.00% | 13.92% (`layers.0.attention.wo`) |

Norms are **never quantized** — 704 values, 2.8 KB at 16.16. Their measured
error is 7.63e-06, exactly the 16.16 rounding floor of 0.5/65536, which is a
tidy confirmation that the fixed-point conversion does what it claims.

**int8 is effectively lossless. int4 is a real ~12% perturbation, and the
drawings survive it** at the temperature actually used. int8 would be ~2×
slower for no visible gain.

A trap worth naming: at temperature 0 the int4 output *looked better* than
fp32, and the config that looked best had the **highest** error of the three.
There is no mechanism by which a 12% weight perturbation improves a model.
Greedy decoding is a knife-edge — one shifted logit flips an argmax and the
whole trajectory cascades. Excellent as a determinism check, useless for
ranking configurations. Judge quantization at the temperature you'll actually
use, not at zero.

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
`output` if the classifier isn't tied. Every weight row is padded to a multiple
of `group`, so row *i* starts at element `i*stride` and its scales at `i*gpr`,
both group-aligned. Without that padding, `w2` (64×172) would have groups
straddling row boundaries and the matmul would need to track two unaligned
group indices at once.

**int4 packing:** two's-complement nibbles in [−8, 7]. Element 2i in the LOW
nibble of byte i, element 2i+1 in the HIGH nibble. Odd tail zero-padded. The C
unpacker must match exactly — `verify_psk.py:unpack_int4` is the reference.

The file is little-endian, which is the wrong endianness for a 68000. That's
deliberate — it stays inspectable on the host, and `core/psk.c` reads every
multi-byte field byte by byte.

## Macintosh Plus budget

Under System 6 without MultiFinder the application heap grows into all free
RAM, which on a 1 MB Plus is roughly 850 KB.

| | |
|---|---|
| `plus_sketch_q4.psk` (weights + scales + norms + codebook) | 158 KB |
| runtime state — KV cache 140 KB, activations, RoPE tables, logits | 157 KB |
| application | 24 KB |
| **total** | **~340 KB** |

Most of the runtime state is the KV cache, stored as int32 rather than int8 —
worth 105 KB if it ever needs reclaiming, at the cost of extra quantization
error. It isn't close to binding, so it stays simple.

Memory was never the constraint. **int4 is a speed decision.**

## Porting notes

Four bugs in the fixed-point port produced no crash, no warning, and entirely
plausible output. Each took a while to find.

### `lo * (int32_t)xq[]` cost 665 cycles per MAC

The matmul inner loop looked cheap — int4 weights against int8 activations —
but `lo` was declared `int` and the activation was cast to `int32_t`, making it
a **32×32 multiply**. The 68000 has no `MULS.L` until the 68020, so GCC emitted
a `__mulsi3` helper call for every multiply-accumulate. Dividing the measured
token time by the MAC count gave 665 cycles each, which is a subroutine call,
not an instruction.

The fix removes multiplication from the inner loop entirely. The loops are
inverted so activations are outermost and output rows inner; for a fixed pair
of activations there are only 16 possible int4 weight values, so two 16-entry
tables hold every product that can occur — and they are built by repeated
*addition* (`lot[k+1] = lot[k] + x0`), so even the table construction doesn't
multiply. Each weight byte then costs two lookups and an add.

Cost: the accumulator must live in memory rather than a register, and the
weight walk is strided. Neither matters on a machine with no cache.

**30.11 → 1.89 s/token.**

The same mistake, in a smaller form, was in attention: `fx_mul` needs a 32×32→64
product and so is also a helper call, ~14,000 times per token. Shifting both
operands to 12.4 first lets a single `MULS.W` do it.

### Activation scales need 8.24, not 16.16

Activations are small — max |x| around 0.005 is normal — so an integer 16.16
scale has two or three representable steps and throws away 20%+ of the scale.
Weight scales are fine at 16.16 because weights are larger.

The first attempt at the attention shift used 8.8 for the same reason and was
wrong in the same way: attention weights at long context are around 1/60, which
in 8.8 rounds to 2 or 3. 12.4 keeps 16× more.

### The matmul must carry the scales through one int64 chain

Combining the two scales first with `fx_mul` looks tidier and loses just as
much: their product is often a handful of 16.16 units (0.04 × 0.001 → 2.6,
stored as 2). Fixing this halved the logit error.

### The sampler must scale by temperature before masking

Masking category tokens to `INT32_MIN/2` and *then* multiplying by `1/T`
overflows int32 for temperatures roughly between 0.25 and 0.4, wrapping the
sentinel positive so every masked token becomes the argmax. The symptom was a
Macintosh that appeared to hang at T=0.3 while 0.2 and 0.5 worked fine — it was
generating category tokens, which produce no strokes, until the sequence limit.

### RoPE frequencies need 8.24 as well

The slowest rotation channels are around 0.0056, only 368 units of 16.16.
Truncating there costs 0.1% on the frequency, which is then multiplied by the
position — so by position 100 the sine is 3.5% low. Computing the frequency at
8.24 brings that to 0.01%.

### The gate that found all of them

`training/dump_ref.py` runs the *quantized* weights through PyTorch and dumps
logits at every position; `host.c` diffs the C against them.

```bash
python dump_ref.py --psk=../models/plus_sketch_q4.psk \
    --out=../tests/ref_q4.bin --quant_acts
make && ./host models/plus_sketch_q4.psk tests/ref_q4.bin
```

`--quant_acts` matters. Without it the reference keeps activations in float
while the C quantizes them, and the C looks far worse than it is.

Position 0 matches to 1e-4, which verifies embedding, RMSNorm, all 36 matmuls,
int4 unpacking, group scales, SwiGLU and the output projection. Later positions
drift by up to 0.5 on logits spanning ±12 — fixed-point accumulation through
RoPE and the KV cache, well below what int4's 12% weight error contributes.

Its blind spot is worth naming: the reference is 12 tokens long, so it never
exercises long context. The 8.8 attention bug passed it cleanly and only showed
up as a bad drawing at position 60.

**Given the same seed, the Macintosh and the macOS build produce bit-identical
drawings.** Same sampler, same arithmetic, opposite endianness. That is the
strongest evidence the port is correct — not "looks similar", identical.

## Scripts

Everything under `training/` runs from inside that directory. llama2.c uses
flat imports and CWD-relative paths.

| script | what it does | when to run it |
|---|---|---|
| `sketchdata.py` | the data pipeline — four subcommands, below | building a dataset |
| `sketchtask.py` | dataset loader; replaces llama2.c's `tinystories.Task` | never directly, `train.py` imports it |
| `train.py` | training loop, from llama2.c (one line changed: the `Task` import) | once there are `.bin` shards |
| `sample.py` | draw from an fp32 checkpoint; renders PNG and SVG | evaluating a model, choosing temperature |
| `quantsweep.py` | fake-quantizes in memory and draws, across bit widths and group sizes | **before** writing any C |
| `export_fixed.py` | writes `.psk` files: int4, int8, or fp32 | once the model is good enough |
| `verify_psk.py` | reads a `.psk` back, checks it, samples from it | after every export; also the spec for the C loader |
| `dump_ref.py` | runs the quantized weights through PyTorch, dumps logits per position | to check the C against something |
| `scripts/make_disk.sh` | builds `dist/PlusSketch.dsk` from the current 68k build | after any 68k rebuild |

### `sketchdata.py` subcommands

- **`fetch`** — download Quick, Draw! sketch-rnn `.npz` files (~6 GB for all 345).
- **`codebook`** — k-means over pen deltas, writes `models/codebook.npz`. Runs
  once; the result is independent of length filters, so it never needs
  rebuilding when re-tokenizing.
- **`roundtrip`** — renders real drawings above their quantized versions.
  **This is the gate.** The bottom row *is* the training data and it is the
  model's ceiling; if it isn't recognizable, no amount of training will help.
- **`tokenize`** — `.npz` → `.bin` shards. Cheap and disposable; re-run freely
  to change `max_points` or `max_seq_len`.

### The C

| file | what |
|---|---|
| `core/fixed.h` | 16.16 arithmetic; `mul16` is the only cheap multiply on a 68000 |
| `core/tables.h` | 1.3 KB of exp/sin lookup data; generated, don't edit |
| `core/psk.c` | endian-safe `.psk` loader |
| `core/forward.c` | the transformer — int8 activations × int4 weights, no multiply in the inner loop |
| `core/sketch.c` | tokens → strokes, plus the degeneracy guard |
| `platform/host.c` | macOS driver: SVG output, reference comparison |
| `platform/mac68k.c` | Macintosh: QuickDraw UI, File Manager, typed prompt |
| `mt.c` | prints the magnitude of each stage after one forward pass; finds which one collapsed |

`core/` links no libm, allocates nothing it doesn't free, and contains no
floating point. It compiles identically on macOS and 68k.

### Building for the Mac

The Retro68 Docker image avoids building a cross-compiler from source, which
several people report fighting with on Apple Silicon.

```bash
docker run --rm -v $(pwd):/root ghcr.io/autc04/retro68 bash -c \
 "cd /root && rm -rf build-68k && mkdir build-68k && cd build-68k && \
  cmake .. -DCMAKE_TOOLCHAIN_FILE=/Retro68-build/toolchain/m68k-apple-macos/cmake/retro68.toolchain.cmake && \
  make"
./scripts/make_disk.sh
```

Two things that bit: RetroConsole is C++, so linking it needs
`project(... CXX)` or the C++ runtime symbols go undefined — but it is not
used any more, since QuickDraw from ROM is both smaller and closer to the
final design. And the multiversal interfaces don't define the old `monaco`
font constant; use the numeric font ID 4.

## Learning moments

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
| cat | 69 | 0.2–0.3 |
| bicycle | 68 | 0.15 |
| octopus | 72 | 0.2 |

Longer drawings generally want lower temperature — more tokens for error to
compound through — but length alone doesn't predict it. Cat and bicycle are the
same length and differ by a factor of two, because a cat is organic (a wobbly
ear is still an ear) while a bicycle is rigid (two similar wheels joined at
specific points).

**int4 wants lower temperature than fp32 did.** Quantization noise perturbs
every logit, which behaves like added sampling temperature, so the effective T
is higher than the dialled value. Cat peaked at 0.3 in PyTorch and at 0.2 in
the int4 C. Tune on the model that actually runs, not the one you trained.

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

**Training is launch-bound, not compute-bound.** At batch 128 an M5 Max
(40-core GPU) sat at ~50% utilization while achieving 2.3% of peak FLOPs and
~1% of memory bandwidth. A 5-layer transformer forward+backward is ~300
individual kernel dispatches; at ~100 µs each that's the entire iteration time.
Raising batch size to 256 does the same number of dispatches with more work
each — but also does fewer optimizer updates per token, which matters, so 256
rather than 512.

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

python sample.py --checkpoint=out/ckpt.pt --category=cat --n=5 --png --temperature=0.3
python quantsweep.py --checkpoint=out/ckpt.pt --category=bicycle
python export_fixed.py --checkpoint=out/ckpt.pt --bits=4
python verify_psk.py --psk=../models/plus_sketch_q4.psk --draw=cat
```

1.58M drawings, 177M tokens, 20 epochs ≈ 13,000 tokens per parameter, about two
hours on an M5 Max with a 40-core GPU.

### Design notes

- Drawings are padded to exactly `max_seq_len`, because llama2.c's dataset takes
  *aligned* windows. One window = one complete drawing, never a slice across
  two. Costs ~50% of tokens to padding; worth it.
- Category tokens are masked out of the logits after position 1, or the model
  starts drawing a house halfway through a cat.
- Category indices are **zero-based**. `cat` is line 65 of `categories.txt` and
  index 64.
- Typing, never a menu. A menu implies stored clip art being traced; typing
  implies generation. Unknown words get `I DON'T KNOW HOW TO DRAW A ___.` —
  the honest decline is a feature.
- The drawing streams as it generates, via `sketch_generate`'s `on_point`
  callback, with the bounding box rescaling as it grows. On a machine this slow
  that is not a nicety.

## Hardware compatibility

Built for the **68000**, so it runs on everything upward: Plus, SE, Classic,
Portable, and all later 68k Macs. The reverse isn't true — a 68020+ binary
won't run on a Plus.

The 68020 added 32-bit addressing (the 68000 only wires 24 lines, capping it at
16 MB), scaled index addressing, an instruction cache, and a much faster
multiply. The 68030 adds an MMU and data cache. A **Mac SE/30** should manage
roughly **0.7 s/token** against the Plus's 1.89.

## Roadmap

- [x] Data pipeline: fetch, codebook, round-trip validation, tokenize
- [x] Training and macOS sampling
- [x] Quantization validated in PyTorch (int4, group 64)
- [x] `.psk` export and verification (int4, int8, fp32)
- [x] `core/` — pure-integer C, no float anywhere, verified against PyTorch
- [x] Retro68 cross-compile, Mini vMac
- [x] QuickDraw rendering, typed prompt, temperature control
- [x] 30.11 → 1.89 s/token
- [ ] Degeneracy loops — most categories run to the sequence limit emitting a
      repeating stroke. The guard in `sketch.c` catches short cycles but not
      whatever `house` and `cake` fall into; a stroke-length cap may be the
      better shape of fix
- [ ] Real hardware

## Prior art

The tokenization here is essentially **Sketchformer** (Ribeiro, Bui, Collomosse
& Ponti, CVPR 2020), which built a k-means dictionary over `(Δx, Δy)` with
K=1000 and deliberately oversampled inter-stroke transitions — the same problem
the two-codebook split solves differently. **SketchGPT** (Tiwari, Biswas &
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