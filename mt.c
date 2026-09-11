/* mt.c -- isolate which stage broke.
 *   clang -O2 -o mt mt.c core/psk.c core/forward.c -Icore -lm
 *   ./mt models/plus_sketch_q4.psk
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "forward.h"

int main(int argc, char **argv)
{
    psk_model m; psk_state s; const char *e = 0;
    const char *path = argc > 1 ? argv[1] : "models/plus_sketch_q4.psk";
    int i;
    fx_t mx;

    if (psk_load(&m, path, &e)) { fprintf(stderr, "%s\n", e); return 1; }
    if (psk_state_init(&s, &m.cfg)) { fprintf(stderr, "state\n"); return 1; }

    /* embedding for BOS */
    psk_forward(&m, &s, TOK_BOS, 0);

    printf("after one forward pass at pos 0:\n");

    mx = 0;
    for (i = 0; i < m.cfg.dim; i++) { fx_t a = s.x[i] < 0 ? -s.x[i] : s.x[i]; if (a > mx) mx = a; }
    printf("  residual x   max |.| = %10.5f\n", (double)mx / FX_ONE);

    mx = 0;
    for (i = 0; i < m.cfg.dim; i++) { fx_t a = s.q[i] < 0 ? -s.q[i] : s.q[i]; if (a > mx) mx = a; }
    printf("  q (last lyr) max |.| = %10.5f\n", (double)mx / FX_ONE);

    mx = 0;
    for (i = 0; i < m.cfg.hidden_dim; i++) { fx_t a = s.hb[i] < 0 ? -s.hb[i] : s.hb[i]; if (a > mx) mx = a; }
    printf("  hb  (SwiGLU) max |.| = %10.5f\n", (double)mx / FX_ONE);

    mx = 0;
    for (i = 0; i < m.cfg.vocab_size; i++) { fx_t a = s.logits[i] < 0 ? -s.logits[i] : s.logits[i]; if (a > mx) mx = a; }
    printf("  logits       max |.| = %10.5f   (expect ~8)\n", (double)mx / FX_ONE);

    printf("\n  first 6 kcache entries (layer 0, pos 0):\n    ");
    for (i = 0; i < 6; i++) printf("%.5f ", (double)s.kcache[i] / FX_ONE);
    printf("\n");

    printf("  first 6 activation scales xs[]: ");
    for (i = 0; i < 2; i++) printf("%ld ", (long)s.xs[i]);
    printf("\n");

    psk_state_free(&s); psk_free(&m);
    return 0;
}
