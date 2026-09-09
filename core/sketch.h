/* sketch.h -- turn a token stream into pen movements.
 *
 * This is the whole decoder. On the Mac it drives QuickDraw MoveTo/LineTo;
 * on the host it writes SVG. No neural network on the output side -- just a
 * 1.5 KB codebook lookup, which is why this fits a 68000.
 */

#ifndef PSK_SKETCH_H
#define PSK_SKETCH_H

#include "forward.h"

#define SKETCH_MAX_PTS 512

typedef struct {
    int32_t x[SKETCH_MAX_PTS];
    int32_t y[SKETCH_MAX_PTS];
    int16_t start[SKETCH_MAX_PTS];  /* index where each stroke begins */
    int     n_pts;
    int     n_strokes;
    int     n_tokens;
    int     hit_eos;
} sketch;

/* Generate one drawing. inv_temp is 1/T in 16.16; 0 means greedy.
 * Calls on_point(ctx, x, y, pen_down) after each point if non-NULL, so the
 * Mac can draw as it goes rather than waiting ~3 minutes for the end. */
void sketch_generate(const psk_model *m, psk_state *s, sketch *sk,
                     int category, fx_t inv_temp, uint32_t *rng,
                     void (*on_point)(void *, int32_t, int32_t, int),
                     void *ctx);

/* Scale strokes to fit a size x size box with the given margin. */
void sketch_fit(sketch *sk, int32_t size, int32_t margin);

#endif /* PSK_SKETCH_H */
