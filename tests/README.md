# Test fixtures

`ref_q4.bin` — per-position logits from `models/plus_sketch_q4.psk`, run
through PyTorch with activation quantization matching `core/forward.c`.

    make && ./host models/plus_sketch_q4.psk tests/ref_q4.bin

Regenerate after any re-export or retrain:

    cd training
    python dump_ref.py --psk=../models/plus_sketch_q4.psk \
        --out=../tests/ref_q4.bin --quant_acts

`--quant_acts` is not optional. Without it the reference keeps activations in
float while the C quantizes them, and the C looks far worse than it is.
