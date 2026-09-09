/* forward.h -- runtime state and the transformer forward pass.
 *
 * All arithmetic is integer. Activations are quantized to int8 per group so
 * the matmul inner loop is int8 x int4 accumulated into int32 -- no shifts,
 * no scaling per element. Scales are applied once per group of 64.
 */

#ifndef PSK_FORWARD_H
#define PSK_FORWARD_H

#include "psk.h"

typedef struct {
    const psk_config *cfg;

    fx_t *x;        /* dim -- the residual stream */
    fx_t *xb;       /* dim */
    fx_t *xb2;      /* dim */
    fx_t *hb;       /* hidden_dim */
    fx_t *hb2;      /* hidden_dim */
    fx_t *q;        /* dim */
    fx_t *att;      /* n_heads * max_seq_len */
    fx_t *logits;   /* vocab_size */

    int8_t *xq;     /* quantized dim-vector, padded to a group multiple */
    fx_t   *xs;
    int8_t *hq;     /* quantized hidden-vector */
    fx_t   *hs;

    fx_t *kcache;   /* n_layers * max_seq_len * kv_dim */
    fx_t *vcache;

    fx_t *rope_c;   /* max_seq_len * (head_dim/2) */
    fx_t *rope_s;

    fx_t inv_sqrt_hd;
} psk_state;

int  psk_state_init(psk_state *s, const psk_config *cfg);
void psk_state_free(psk_state *s);
long psk_state_bytes(const psk_config *cfg);

/* Runs one token at position pos, updating the KV cache.
 * Returns s->logits (vocab_size entries, 16.16). */
const fx_t *psk_forward(const psk_model *m, psk_state *s, int token, int pos);

/* Temperature sampling over logits. rng_state is xorshift, seeded non-zero.
 * inv_temp is 1/T in 16.16; pass 0 for greedy (argmax).
 * Category tokens are masked when pos > 0. */
int psk_sample(const psk_model *m, fx_t *logits, fx_t inv_temp,
               int pos, uint32_t *rng_state);

#endif /* PSK_FORWARD_H */
