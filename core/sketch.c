/* sketch.c -- decode a token stream into strokes.
 *
 * Mirrors training/sketchdata.py:decode(). A drawing starts at the origin
 * with the pen down; a draw token extends the current stroke; a jump token
 * starts a new one; PEN_UP marks the end of a stroke.
 */

#include <string.h>

#include "sketch.h"

static void push(sketch *sk, int32_t x, int32_t y, int new_stroke,
                 void (*on_point)(void *, int32_t, int32_t, int), void *ctx)
{
    if (sk->n_pts >= SKETCH_MAX_PTS) return;
    if (new_stroke && sk->n_strokes < SKETCH_MAX_PTS)
        sk->start[sk->n_strokes++] = (int16_t)sk->n_pts;
    sk->x[sk->n_pts] = x;
    sk->y[sk->n_pts] = y;
    sk->n_pts++;
    if (on_point) on_point(ctx, x, y, !new_stroke);
}

void sketch_generate(const psk_model *m, psk_state *s, sketch *sk,
                     int category, fx_t inv_temp, uint32_t *rng,
                     void (*on_point)(void *, int32_t, int32_t, int),
                     void *ctx)
{
    const psk_config *c = &m->cfg;
    int32_t x = 0, y = 0;
    int pos = 0, tok, new_stroke = 1;
    int last_tok = -1, prev_tok = -1, run = 0;

    memset(sk, 0, sizeof(*sk));
    push(sk, 0, 0, 1, on_point, ctx);
    new_stroke = 0;

    /* prime with BOS and the category token */
    psk_forward(m, s, TOK_BOS, pos++);
    tok = c->cat_base + category;
    psk_forward(m, s, tok, pos++);

    while (pos < c->max_seq_len) {
        fx_t *lg = s->logits;
        int16_t dx, dy;
        int kind;

        tok = psk_sample(m, lg, inv_temp, pos, rng);
        sk->n_tokens++;

        if (tok == TOK_EOS) { sk->hit_eos = 1; break; }

        /* Degeneracy guard. A single token repeating, or two alternating,
         * is never a real drawing -- it is the model stuck in a cycle. The
         * alternating case shows up as a dashed diagonal line, because it
         * is an offset and a PEN_UP taking turns. */
        if (tok == last_tok || tok == prev_tok) {
            if (++run >= 8) break;
        } else {
            run = 0;
        }
        prev_tok = last_tok;
        last_tok = tok;

        if (tok == TOK_PEN_UP) {
            new_stroke = 1;
        } else {
            kind = psk_offset(m, tok, &dx, &dy);
            if (kind) {
                x += dx;
                y += dy;
                push(sk, x, y, new_stroke || kind == 2, on_point, ctx);
                new_stroke = 0;
            }
        }

        psk_forward(m, s, tok, pos);
        pos++;
    }
    if (sk->n_strokes == 0 && sk->n_pts > 0)
        sk->start[sk->n_strokes++] = 0;
}

void sketch_fit(sketch *sk, int32_t size, int32_t margin)
{
    int32_t minx, maxx, miny, maxy, w, h, span, box;
    int i;

    if (sk->n_pts < 2) return;
    minx = maxx = sk->x[0];
    miny = maxy = sk->y[0];
    for (i = 1; i < sk->n_pts; i++) {
        if (sk->x[i] < minx) minx = sk->x[i];
        if (sk->x[i] > maxx) maxx = sk->x[i];
        if (sk->y[i] < miny) miny = sk->y[i];
        if (sk->y[i] > maxy) maxy = sk->y[i];
    }
    w = maxx - minx; if (w < 1) w = 1;
    h = maxy - miny; if (h < 1) h = 1;
    span = (w > h) ? w : h;
    box = size - 2 * margin;

    /* integer scale: (v - min) * box / span, no division per point beyond
     * this one ratio */
    for (i = 0; i < sk->n_pts; i++) {
        sk->x[i] = margin + (int32_t)(((int64_t)(sk->x[i] - minx) * box) / span);
        sk->y[i] = margin + (int32_t)(((int64_t)(sk->y[i] - miny) * box) / span);
    }
}
