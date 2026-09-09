# Model artifacts

| file | what |
|---|---|
| `codebook.npz` | the tokenizer — 256 draw + 128 jump offset vectors, 4 KB |
| `plus_sketch_ckpt.pt` | fp32 training checkpoint; the only thing you can fine-tune from |
| `plus_sketch_fp32.psk` | fp32 weights in the deployable format |
| `plus_sketch_q8.psk` | int8, group 64 — near-lossless, ~2.9x slower on 68k |
| `plus_sketch_q4.psk` | **int4, group 64** |

## Config

```
dim=64 n_layers=5 n_heads=4 n_kv_heads=2 hidden_dim=172
vocab_size=732 max_seq_len=112
274,112 parameters
```

## Training

```
python sketchdata.py tokenize --vocab_size=732 --max_points=96
--max_seq_len=112 --per_category=5000
python train.py --out_dir=out --vocab_source=custom --vocab_size=732
--dim=64 --n_layers=5 --n_heads=4 --n_kv_heads=2 --multiple_of=4
--max_seq_len=112 --batch_size=256 --gradient_accumulation_steps=1
--learning_rate=1.5e-3 --dropout=0.05 --weight_decay=0.01 --beta2=0.99
--warmup_iters=1000 --max_iters=123819
--eval_interval=2000 --eval_iters=100
--device=mps --dtype=float32 --compile=False
```


Quick, Draw!, all 345 categories, 5,000 drawings each, filtered to ≤96 points.
1.58M drawings, 177M tokens, 20 epochs (~13,000 tokens/parameter), ~2 hours
on an M5 Max 40-core GPU.

Final validation loss: **TODO**

## Quantization error

| | mean | worst tensor |
|---|---|---|
| int8, group 64 | 0.66% | 0.78% (`layers.0.attention.wo`) |
| int4, group 64 | 12.00% | 13.92% (`layers.0.attention.wo`) |

Norms are never quantized — 704 values at 16.16, error 7.63e-06, exactly the
format's rounding floor.

## Recommended temperature

**0.20** for the int4 model. Note this is *lower* than the 0.3 that was optimal
for fp32: quantization noise perturbs every logit, which behaves like added
sampling temperature, so the effective T is higher than the dialled value.
Tune on the model you ship.

Category indices are **zero-based** — `cat` is line 65 of `categories.txt`
and index 64.
