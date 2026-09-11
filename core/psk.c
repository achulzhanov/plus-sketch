/* psk.c -- load a .psk model file.
 *
 * Mirrors training/verify_psk.py:read_psk() exactly. If that consumes every
 * byte, this must too -- the final check in psk_load() enforces it, which
 * catches tensor-order mistakes loudly instead of loading garbage.
 */

#include <stdlib.h>
#include <string.h>
#ifndef PSK_NO_STDIO
#include <stdio.h>
#endif

#include "psk.h"

/* ---- little-endian readers -------------------------------------------
 * The file is LE; the 68000 is BE. Read byte by byte, always. */

static int32_t rd_i32(const uint8_t *p)
{
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static int16_t rd_i16(const uint8_t *p)
{
    return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

/* Convert an LE int32 array in place to host order. Returns the same
 * pointer, typed. Safe because sizeof(int32_t) == 4 on every target here. */
static int32_t *fix_i32(uint8_t *p, int32_t n)
{
    int32_t i;
    for (i = 0; i < n; i++) {
        int32_t v = rd_i32(p + i * 4);
        ((int32_t *)p)[i] = v;
    }
    return (int32_t *)p;
}

static int16_t *fix_i16(uint8_t *p, int32_t n)
{
    int32_t i;
    for (i = 0; i < n; i++) {
        int16_t v = rd_i16(p + i * 2);
        ((int16_t *)p)[i] = v;
    }
    return (int16_t *)p;
}

/* ---- cursor over the blob -------------------------------------------- */

typedef struct {
    uint8_t *base;
    long     size;
    long     pos;
    int      overflow;
} cursor;

static uint8_t *take(cursor *c, long n)
{
    uint8_t *p;
    if (c->pos + n > c->size) { c->overflow = 1; return NULL; }
    p = c->base + c->pos;
    c->pos += n;
    return p;
}

/* Read one quantized tensor: scales then values. */
static int read_tensor(cursor *c, psk_tensor *t, const psk_config *cfg,
                       int32_t rows, int32_t cols)
{
    int32_t gpr, stride, n, n_scales, n_bytes;
    uint8_t *p;

    t->rows = rows; t->cols = cols;

    if (cfg->bits == 32) {
        t->stride = cols; t->gpr = 0; t->n = rows * cols; t->scales = NULL;
        p = take(c, (long)rows * cols * 4);
        if (!p) return -1;
        t->q = (const int8_t *)p;
        return 0;
    }

    gpr    = (cols + cfg->group - 1) / cfg->group;
    stride = gpr * cfg->group;
    n      = rows * stride;
    t->gpr = gpr; t->stride = stride; t->n = n;

    n_scales = rows * gpr;
    p = take(c, (long)n_scales * 4);
    if (!p) return -1;
    t->scales = (const fx_t *)fix_i32(p, n_scales);

    n_bytes = (cfg->bits == 8) ? n : (n + 1) / 2;
    p = take(c, n_bytes);
    if (!p) return -1;
    t->q = (const int8_t *)p;
    return 0;
}

/* ---------------------------------------------------------------------- */

int psk_load_mem(psk_model *m, void *blob_in, long size, const char **err)
{
    uint8_t *blob = (uint8_t *)blob_in, *p;
    cursor c;
    psk_config *cfg = &m->cfg;
    int32_t i, L;

    memset(m, 0, sizeof(*m));
    if (err) *err = NULL;
    m->blob = blob;

    if (size < PSK_HEADER) {
        if (err) *err = "file too small";
        goto fail;
    }
    c.base = blob; c.size = size; c.pos = 0; c.overflow = 0;

    p = take(&c, PSK_HEADER);
    if (memcmp(p, PSK_MAGIC, 4) != 0) {
        if (err) *err = "bad magic (not a .psk file)";
        goto fail;
    }
    if (rd_i32(p + 4) != PSK_VERSION) {
        if (err) *err = "unsupported version";
        goto fail;
    }
    cfg->bits        = rd_i32(p + 8);
    cfg->group       = rd_i32(p + 12);
    cfg->dim         = rd_i32(p + 16);
    cfg->hidden_dim  = rd_i32(p + 20);
    cfg->n_layers    = rd_i32(p + 24);
    cfg->n_heads     = rd_i32(p + 28);
    cfg->n_kv_heads  = rd_i32(p + 32);
    cfg->vocab_size  = rd_i32(p + 36);
    cfg->max_seq_len = rd_i32(p + 40);
    cfg->shared      = rd_i32(p + 44);
    cfg->n_draw      = rd_i32(p + 48);
    cfg->n_jump      = rd_i32(p + 52);

    cfg->head_dim  = cfg->dim / cfg->n_heads;
    cfg->kv_dim    = cfg->head_dim * cfg->n_kv_heads;
    cfg->draw_base = TOK_DRAW;
    cfg->jump_base = cfg->draw_base + cfg->n_draw;
    cfg->cat_base  = cfg->jump_base + cfg->n_jump;

    if (cfg->bits != 4 && cfg->bits != 8 && cfg->bits != 32) {
        if (err) *err = "bits must be 4, 8 or 32";
        goto fail;
    }

    p = take(&c, (long)cfg->n_draw * 4);
    if (!p) goto trunc;
    m->draw_cb = fix_i16(p, cfg->n_draw * 2);
    p = take(&c, (long)cfg->n_jump * 4);
    if (!p) goto trunc;
    m->jump_cb = fix_i16(p, cfg->n_jump * 2);

    L = cfg->n_layers;
    {
        fx_t *att = (fx_t *)malloc(sizeof(fx_t) * L * cfg->dim);
        fx_t *ffn = (fx_t *)malloc(sizeof(fx_t) * L * cfg->dim);
        if (!att || !ffn) {
            free(att); free(ffn);
            if (err) *err = "out of memory";
            goto fail;
        }
        for (i = 0; i < L; i++) {
            p = take(&c, (long)cfg->dim * 4);
            if (!p) { free(att); free(ffn); goto trunc; }
            memcpy(att + i * cfg->dim, fix_i32(p, cfg->dim),
                   sizeof(fx_t) * cfg->dim);
            p = take(&c, (long)cfg->dim * 4);
            if (!p) { free(att); free(ffn); goto trunc; }
            memcpy(ffn + i * cfg->dim, fix_i32(p, cfg->dim),
                   sizeof(fx_t) * cfg->dim);
        }
        m->att_norm = att;
        m->ffn_norm = ffn;
    }
    p = take(&c, (long)cfg->dim * 4);
    if (!p) goto trunc;
    m->final_norm = (const fx_t *)fix_i32(p, cfg->dim);

    m->wq = (psk_tensor *)calloc((size_t)L, sizeof(psk_tensor));
    m->wk = (psk_tensor *)calloc((size_t)L, sizeof(psk_tensor));
    m->wv = (psk_tensor *)calloc((size_t)L, sizeof(psk_tensor));
    m->wo = (psk_tensor *)calloc((size_t)L, sizeof(psk_tensor));
    m->w1 = (psk_tensor *)calloc((size_t)L, sizeof(psk_tensor));
    m->w2 = (psk_tensor *)calloc((size_t)L, sizeof(psk_tensor));
    m->w3 = (psk_tensor *)calloc((size_t)L, sizeof(psk_tensor));
    if (!m->wq || !m->wk || !m->wv || !m->wo || !m->w1 || !m->w2 || !m->w3) {
        if (err) *err = "out of memory";
        goto fail;
    }

    if (read_tensor(&c, &m->tok_emb, cfg, cfg->vocab_size, cfg->dim)) goto trunc;
    for (i = 0; i < L; i++) {
        if (read_tensor(&c, &m->wq[i], cfg, cfg->dim,        cfg->dim))        goto trunc;
        if (read_tensor(&c, &m->wk[i], cfg, cfg->kv_dim,     cfg->dim))        goto trunc;
        if (read_tensor(&c, &m->wv[i], cfg, cfg->kv_dim,     cfg->dim))        goto trunc;
        if (read_tensor(&c, &m->wo[i], cfg, cfg->dim,        cfg->dim))        goto trunc;
        if (read_tensor(&c, &m->w1[i], cfg, cfg->hidden_dim, cfg->dim))        goto trunc;
        if (read_tensor(&c, &m->w2[i], cfg, cfg->dim,        cfg->hidden_dim)) goto trunc;
        if (read_tensor(&c, &m->w3[i], cfg, cfg->hidden_dim, cfg->dim))        goto trunc;
    }
    if (cfg->shared)
        m->out = m->tok_emb;
    else if (read_tensor(&c, &m->out, cfg, cfg->vocab_size, cfg->dim))
        goto trunc;

    if (c.pos != c.size) {
        if (err) *err = "trailing bytes -- tensor order does not match the exporter";
        goto fail;
    }
    return 0;

trunc:
    if (err) *err = "file truncated -- tensor order does not match the exporter";
fail:
    psk_free(m);
    return -1;
}

#ifndef PSK_NO_STDIO
int psk_load(psk_model *m, const char *path, const char **err)
{
    FILE *f;
    long size;
    uint8_t *blob;

    memset(m, 0, sizeof(*m));
    if (err) *err = NULL;

    f = fopen(path, "rb");
    if (!f) { if (err) *err = "cannot open file"; return -1; }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    blob = (uint8_t *)malloc((size_t)size);
    if (!blob) { fclose(f); if (err) *err = "out of memory"; return -1; }
    if (fread(blob, 1, (size_t)size, f) != (size_t)size) {
        free(blob); fclose(f); if (err) *err = "short read"; return -1;
    }
    fclose(f);
    return psk_load_mem(m, blob, size, err);
}
#endif

void psk_free(psk_model *m)
{
    if (!m) return;
    free((void *)m->att_norm);
    free((void *)m->ffn_norm);
    free(m->wq); free(m->wk); free(m->wv); free(m->wo);
    free(m->w1); free(m->w2); free(m->w3);
    free(m->blob);
    memset(m, 0, sizeof(*m));
}

int psk_offset(const psk_model *m, int tok, int16_t *dx, int16_t *dy)
{
    const psk_config *c = &m->cfg;
    if (tok >= c->draw_base && tok < c->jump_base) {
        const int16_t *e = m->draw_cb + (tok - c->draw_base) * 2;
        *dx = e[0]; *dy = e[1];
        return 1;
    }
    if (tok >= c->jump_base && tok < c->cat_base) {
        const int16_t *e = m->jump_cb + (tok - c->jump_base) * 2;
        *dx = e[0]; *dy = e[1];
        return 2;   /* a jump: starts a new stroke */
    }
    return 0;
}

#ifndef PSK_NO_STDIO
void psk_print_config(const psk_model *m)
{
    const psk_config *c = &m->cfg;
    printf("PSK: %ld-bit, group %ld\n", (long)c->bits, (long)c->group);
    printf("  dim=%ld hidden=%ld layers=%ld heads=%ld kv_heads=%ld\n",
           (long)c->dim, (long)c->hidden_dim, (long)c->n_layers,
           (long)c->n_heads, (long)c->n_kv_heads);
    printf("  vocab=%ld seq=%ld shared=%ld\n",
           (long)c->vocab_size, (long)c->max_seq_len, (long)c->shared);
    printf("  head_dim=%ld kv_dim=%ld\n", (long)c->head_dim, (long)c->kv_dim);
    printf("  codebook: %ld draw, %ld jump\n",
           (long)c->n_draw, (long)c->n_jump);
    printf("  token layout: BOS=0 EOS=1 PEN_UP=2 draw=%ld..%ld "
           "jump=%ld..%ld cat=%ld..%ld\n",
           (long)c->draw_base, (long)(c->jump_base - 1),
           (long)c->jump_base, (long)(c->cat_base - 1),
           (long)c->cat_base, (long)(c->vocab_size - 1));
}
#endif /* PSK_NO_STDIO */