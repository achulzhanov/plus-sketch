/* forward.c -- integer-only transformer forward pass.
 *
 * Structure follows llama2.c's runq.c: activations are quantized to int8 per
 * group so the matmul inner loop is plain integer multiply-accumulate, with
 * scales applied once per group rather than per element. Weights are int4
 * (two per byte, low nibble first) or int8.
 *
 * Row layout: every weight row is padded to a multiple of `group`, so row i
 * starts at element i*stride and its scales at i*gpr, both group-aligned.
 * See training/export_fixed.py.
 */

#include <stdlib.h>
#include <string.h>

#include "forward.h"

/* ---- small helpers ---------------------------------------------------- */

static fx_t *alloc_fx(long n)
{
    return (fx_t *)calloc((size_t)n, sizeof(fx_t));
}

/* Quantize a fx_t vector to int8, per group. Writes `stride` bytes
 * (n rounded up to a group multiple); the tail is zeroed.
 *
 * Scales are 8.24, NOT 16.16. Activations are often small -- max |x| around
 * 0.005 is common -- and an integer 16.16 scale would then have only two or
 * three representable steps, throwing away 20%+ of the scale. The extra 8
 * fractional bits are shifted back out in matmul().
 */
static void quantize_vec(const fx_t *x, int n, int group,
                         int8_t *q, fx_t *s)
{
    int gpr = (n + group - 1) / group;
    int g, j;

    for (g = 0; g < gpr; g++) {
        int base = g * group;
        int cnt  = (base + group <= n) ? group : (n - base);
        fx_t maxv = 0;
        int64_t sc;

        for (j = 0; j < cnt; j++) {
            fx_t a = x[base + j];
            if (a < 0) a = -a;
            if (a > maxv) maxv = a;
        }

        sc = (((int64_t)maxv << 8) + 126) / 127;   /* 8.24, rounded up */
        if (sc <= 0) sc = 1;
        s[g] = (fx_t)sc;

        for (j = 0; j < cnt; j++) {
            int64_t xv = (int64_t)x[base + j] << 8;
            int32_t v = (int32_t)((xv >= 0) ? (xv + sc / 2) / sc
                                            : -(((-xv) + sc / 2) / sc));
            q[base + j] = (int8_t)clamp32(v, -127, 127);
        }
        for (; j < group; j++)
            q[base + j] = 0;
    }
}

/* out[rows] = W * x, where x is already quantized. */
static void matmul(fx_t *out, const int8_t *xq, const fx_t *xs,
                   const psk_tensor *w, int group, int bits)
{
    int i, g, j;
    int rows = w->rows, stride = w->stride, gpr = w->gpr;

    for (i = 0; i < rows; i++) {
        const fx_t *ws = w->scales + (long)i * gpr;
        fx_t total = 0;

        for (g = 0; g < gpr; g++) {
            int base = g * group;
            int32_t ival = 0;

            if (bits == 8) {
                const int8_t *wv = w->q + (long)i * stride + base;
                for (j = 0; j < group; j++)
                    ival += (int32_t)wv[j] * (int32_t)xq[base + j];
            } else {
                /* int4: two weights per byte, low nibble first */
                const uint8_t *wb = (const uint8_t *)w->q
                                  + (((long)i * stride + base) >> 1);
                for (j = 0; j < group; j += 2) {
                    uint8_t b = wb[j >> 1];
                    int lo = b & 0x0F;  if (lo > 7) lo -= 16;
                    int hi = b >> 4;    if (hi > 7) hi -= 16;
                    ival += lo * (int32_t)xq[base + j]
                          + hi * (int32_t)xq[base + j + 1];
                }
            }

            /* sum(w*x) = ival * w_scale * x_scale, in 16.16.
             *
             * Do NOT precombine the scales with fx_mul: their product is
             * often only a handful of 16.16 units (ws 0.04 * xs 0.001 ->
             * 2.6, stored as 2), which throws away 20%+ of the scale. Carry
             * full precision through a single 64-bit chain instead. This is
             * one wide multiply per group of 64 MACs, so it stays out of the
             * hot loop.
             */
            total += (fx_t)((((int64_t)ival * ws[g]) * (int64_t)xs[g])
                            >> (FX_SHIFT + 8));
        }
        out[i] = total;
    }
}

/* Dequantize one row of a weight matrix (used for the embedding lookup). */
static void embed_row(fx_t *out, const psk_tensor *t, int row,
                      int group, int bits)
{
    int j;
    const fx_t *sc = t->scales + (long)row * t->gpr;

    for (j = 0; j < t->cols; j++) {
        int32_t qv;
        if (bits == 8) {
            qv = t->q[(long)row * t->stride + j];
        } else {
            long e = (long)row * t->stride + j;
            uint8_t b = ((const uint8_t *)t->q)[e >> 1];
            qv = (e & 1) ? (b >> 4) : (b & 0x0F);
            if (qv > 7) qv -= 16;
        }
        out[j] = (fx_t)(qv * sc[j / group]);
    }
}

static void rmsnorm(fx_t *o, const fx_t *x, const fx_t *w, int n)
{
    int64_t ss = 0;
    fx_t inv;
    int i;

    for (i = 0; i < n; i++)
        ss += (int64_t)x[i] * (int64_t)x[i];
    ss /= n;
    ss >>= FX_SHIFT;          /* back to 16.16 */
    ss += 1;                  /* eps */
    inv = fx_rsqrt((fx_t)ss);

    for (i = 0; i < n; i++)
        o[i] = fx_mul(fx_mul(x[i], inv), w[i]);
}

static void softmax(fx_t *x, int n)
{
    fx_t maxv = x[0], sum = 0;
    int i;

    for (i = 1; i < n; i++)
        if (x[i] > maxv) maxv = x[i];
    for (i = 0; i < n; i++) {
        x[i] = fx_exp_neg(x[i] - maxv);
        sum += x[i];
    }
    if (sum <= 0) sum = 1;
    for (i = 0; i < n; i++)
        x[i] = fx_div(x[i], sum);
}

/* ---- state ------------------------------------------------------------ */

long psk_state_bytes(const psk_config *c)
{
    long dim = c->dim, hid = c->hidden_dim, seq = c->max_seq_len;
    long hd2 = c->head_dim / 2, g = c->group;
    long xstride = ((dim + g - 1) / g) * g;
    long hstride = ((hid + g - 1) / g) * g;

    return (long)sizeof(fx_t) *
           (3 * dim + 2 * hid + dim + c->n_heads * seq + c->vocab_size
            + 2L * c->n_layers * seq * c->kv_dim + 2 * seq * hd2
            + (xstride / g) + (hstride / g))
         + xstride + hstride;
}

int psk_state_init(psk_state *s, const psk_config *c)
{
    long seq = c->max_seq_len, hd2 = c->head_dim / 2;
    long xstride = ((c->dim + c->group - 1) / c->group) * c->group;
    long hstride = ((c->hidden_dim + c->group - 1) / c->group) * c->group;
    int pos, i;

    memset(s, 0, sizeof(*s));
    s->cfg = c;

    s->x      = alloc_fx(c->dim);
    s->xb     = alloc_fx(c->dim);
    s->xb2    = alloc_fx(c->dim);
    s->hb     = alloc_fx(c->hidden_dim);
    s->hb2    = alloc_fx(c->hidden_dim);
    s->q      = alloc_fx(c->dim);
    s->att    = alloc_fx((long)c->n_heads * seq);
    s->logits = alloc_fx(c->vocab_size);
    s->xs     = alloc_fx(xstride / c->group);
    s->hs     = alloc_fx(hstride / c->group);
    s->kcache = alloc_fx((long)c->n_layers * seq * c->kv_dim);
    s->vcache = alloc_fx((long)c->n_layers * seq * c->kv_dim);
    s->rope_c = alloc_fx(seq * hd2);
    s->rope_s = alloc_fx(seq * hd2);
    s->xq     = (int8_t *)calloc((size_t)xstride, 1);
    s->hq     = (int8_t *)calloc((size_t)hstride, 1);

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
        !s->att || !s->logits || !s->xs || !s->hs || !s->kcache ||
        !s->vcache || !s->rope_c || !s->rope_s || !s->xq || !s->hq) {
        psk_state_free(s);
        return -1;
    }

    /* RoPE tables: freq_i = 10000^(-2i/head_dim), angle = pos * freq_i.
     *
     * freq is held at 8.24, not 16.16. At i=7, freq ~ 0.0056, which is only
     * 368 units of 16.16 -- truncating that costs 0.1% on the angle, and at
     * small angles sin(x) ~ x so the error lands straight on the output.
     * The extra 8 bits fix it. */
    for (i = 0; i < hd2; i++) {
        fx_t e = fx_mul(FX_LN10000,
                        fx_div(FX_FROM_INT(2 * i), FX_FROM_INT(c->head_dim)));
        int64_t freq24 = ((int64_t)fx_exp_neg(-e)) << 8;   /* 8.24 */
        for (pos = 0; pos < seq; pos++) {
            fx_t ang = (fx_t)(((int64_t)pos * freq24) >> 8);  /* back to 16.16 */
            s->rope_c[pos * hd2 + i] = fx_cos(ang);
            s->rope_s[pos * hd2 + i] = fx_sin(ang);
        }
    }

    s->inv_sqrt_hd = fx_div(FX_ONE, fx_sqrt(FX_FROM_INT(c->head_dim)));
    return 0;
}

void psk_state_free(psk_state *s)
{
    if (!s) return;
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->att); free(s->logits);
    free(s->xs); free(s->hs);
    free(s->kcache); free(s->vcache);
    free(s->rope_c); free(s->rope_s);
    free(s->xq); free(s->hq);
    memset(s, 0, sizeof(*s));
}

/* ---- forward ---------------------------------------------------------- */

const fx_t *psk_forward(const psk_model *m, psk_state *s, int token, int pos)
{
    const psk_config *c = &m->cfg;
    int dim = c->dim, kv_dim = c->kv_dim, hd = c->head_dim, hd2 = hd / 2;
    int group = c->group, bits = c->bits;
    int kv_mul = c->n_heads / c->n_kv_heads;
    long seq = c->max_seq_len;
    int l, i, h, t, j;

    embed_row(s->x, &m->tok_emb, token, group, bits);

    for (l = 0; l < c->n_layers; l++) {
        long loff = (long)l * seq * kv_dim;
        fx_t *k = s->kcache + loff + (long)pos * kv_dim;
        fx_t *v = s->vcache + loff + (long)pos * kv_dim;

        rmsnorm(s->xb, s->x, m->att_norm + (long)l * dim, dim);
        quantize_vec(s->xb, dim, group, s->xq, s->xs);

        matmul(s->q, s->xq, s->xs, &m->wq[l], group, bits);
        matmul(k,    s->xq, s->xs, &m->wk[l], group, bits);
        matmul(v,    s->xq, s->xs, &m->wv[l], group, bits);

        /* RoPE: rotate pairs of q (and k, up to kv_dim) */
        for (i = 0; i < dim; i += 2) {
            int fi = (i % hd) / 2;
            fx_t fcr = s->rope_c[pos * hd2 + fi];
            fx_t fci = s->rope_s[pos * hd2 + fi];
            int rotn = (i < kv_dim) ? 2 : 1;
            int r;
            for (r = 0; r < rotn; r++) {
                fx_t *vec = (r == 0) ? s->q : k;
                fx_t v0 = vec[i], v1 = vec[i + 1];
                vec[i]     = fx_mul(v0, fcr) - fx_mul(v1, fci);
                vec[i + 1] = fx_mul(v0, fci) + fx_mul(v1, fcr);
            }
        }

        /* attention */
        for (h = 0; h < c->n_heads; h++) {
            const fx_t *qh = s->q + h * hd;
            fx_t *att = s->att + (long)h * seq;
            fx_t *xbh = s->xb + h * hd;
            long khoff = (long)(h / kv_mul) * hd;

            for (t = 0; t <= pos; t++) {
                const fx_t *kh = s->kcache + loff + (long)t * kv_dim + khoff;
                fx_t score = 0;
                for (j = 0; j < hd; j++)
                    score += fx_mul(qh[j], kh[j]);
                att[t] = fx_mul(score, s->inv_sqrt_hd);
            }
            softmax(att, pos + 1);

            for (j = 0; j < hd; j++) xbh[j] = 0;
            for (t = 0; t <= pos; t++) {
                const fx_t *vh = s->vcache + loff + (long)t * kv_dim + khoff;
                fx_t a = att[t];
                for (j = 0; j < hd; j++)
                    xbh[j] += fx_mul(a, vh[j]);
            }
        }

        quantize_vec(s->xb, dim, group, s->xq, s->xs);
        matmul(s->xb2, s->xq, s->xs, &m->wo[l], group, bits);
        for (i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        /* feed-forward: w2( silu(w1(x)) * w3(x) ) */
        rmsnorm(s->xb, s->x, m->ffn_norm + (long)l * dim, dim);
        quantize_vec(s->xb, dim, group, s->xq, s->xs);
        matmul(s->hb,  s->xq, s->xs, &m->w1[l], group, bits);
        matmul(s->hb2, s->xq, s->xs, &m->w3[l], group, bits);

        for (i = 0; i < c->hidden_dim; i++)
            s->hb[i] = fx_mul(fx_mul(s->hb[i], fx_sigmoid(s->hb[i])),
                              s->hb2[i]);

        quantize_vec(s->hb, c->hidden_dim, group, s->hq, s->hs);
        matmul(s->xb, s->hq, s->hs, &m->w2[l], group, bits);
        for (i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    rmsnorm(s->x, s->x, m->final_norm, dim);
    quantize_vec(s->x, dim, group, s->xq, s->xs);
    matmul(s->logits, s->xq, s->xs, &m->out, group, bits);
    return s->logits;
}

/* ---- sampling --------------------------------------------------------- */

static uint32_t xorshift(uint32_t *st)
{
    uint32_t x = *st;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *st = x;
    return x;
}

int psk_sample(const psk_model *m, fx_t *logits, fx_t inv_temp,
               int pos, uint32_t *rng_state)
{
    const psk_config *c = &m->cfg;
    int n = c->vocab_size, i, best = 0;
    fx_t r, cum;

    if (inv_temp > 0) {
        for (i = 0; i < n; i++)
            logits[i] = fx_mul(logits[i], inv_temp);
    }

    /* Mask AFTER scaling. Masking first and then multiplying by inv_temp
     * overflows int32 for temperatures around 0.25-0.4, wrapping the
     * sentinel positive so every category token becomes the argmax. */
    logits[TOK_BOS] = INT32_MIN / 2;
    if (pos > 0)
        for (i = c->cat_base; i < n; i++)
            logits[i] = INT32_MIN / 2;

    if (inv_temp <= 0) {                       /* greedy */
        for (i = 1; i < n; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    softmax(logits, n);

    /* multinomial: walk the cumulative sum. No sort, so no top-p. */
    r = (fx_t)(xorshift(rng_state) >> 16);     /* 0 .. 65535, i.e. [0,1) */
    cum = 0;
    for (i = 0; i < n; i++) {
        cum += logits[i];
        if (cum > r) return i;
    }
    return n - 1;
}
