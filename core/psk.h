/* psk.h -- the plus-sketch model file.
 *
 * Format is written by training/export_fixed.py and documented in the README.
 * training/verify_psk.py is the executable spec: whatever read_psk() does
 * there, psk_load() must do here -- same tensor order, same nibble packing,
 * same 16.16 conversion.
 *
 * The file is little-endian so it stays inspectable on the host. The 68000 is
 * big-endian, so every multi-byte field is read byte by byte. Never memcpy a
 * struct over this data.
 */

#ifndef PSK_H
#define PSK_H

#include <stdint.h>
#include "fixed.h"

#define PSK_MAGIC   "PSK1"
#define PSK_VERSION 2
#define PSK_HEADER  128

/* Fixed vocabulary layout -- must match training/sketchdata.py */
#define TOK_BOS     0
#define TOK_EOS     1
#define TOK_PEN_UP  2
#define TOK_DRAW    3

typedef struct {
    int32_t bits;          /* 4, 8, or 32 */
    int32_t group;         /* weights per scale */
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t max_seq_len;
    int32_t shared;        /* output matrix tied to embeddings */
    int32_t n_draw;
    int32_t n_jump;
    /* derived */
    int32_t head_dim;
    int32_t kv_dim;
    int32_t draw_base;     /* = TOK_DRAW */
    int32_t jump_base;
    int32_t cat_base;
} psk_config;

/* One quantized tensor: scales plus packed values. */
typedef struct {
    const fx_t   *scales;  /* 16.16, rows * gpr entries */
    const int8_t *q;       /* int8 values, or int4 nibbles low-first */
    int32_t       n;       /* rows * stride, including row padding */
    int32_t       rows;
    int32_t       cols;    /* logical columns */
    int32_t       stride;  /* padded columns; row r starts at r*stride */
    int32_t       gpr;     /* groups per row */
} psk_tensor;

typedef struct {
    psk_config cfg;

    /* codebook: 2 int16 per entry, (dx, dy) */
    const int16_t *draw_cb;
    const int16_t *jump_cb;

    /* norms, 16.16, never quantized */
    const fx_t *att_norm;  /* n_layers * dim */
    const fx_t *ffn_norm;  /* n_layers * dim */
    const fx_t *final_norm;/* dim */

    /* weights */
    psk_tensor tok_emb;
    psk_tensor *wq, *wk, *wv, *wo;   /* [n_layers] */
    psk_tensor *w1, *w2, *w3;        /* [n_layers] */
    psk_tensor out;                  /* == tok_emb when cfg.shared */

    void *blob;            /* the whole file; free this and everything goes */
} psk_model;

/* Returns 0 on success, negative on error. On failure *err points at a
 * static description. The model borrows into the loaded blob, so the blob
 * must outlive it -- psk_free() releases both. */
int  psk_load(psk_model *m, const char *path, const char **err);
void psk_free(psk_model *m);

/* Codebook lookup: writes the (dx, dy) offset for a stroke token.
 * Returns 0 if tok is not a stroke token. */
int  psk_offset(const psk_model *m, int tok, int16_t *dx, int16_t *dy);

void psk_print_config(const psk_model *m);

#endif /* PSK_H */
