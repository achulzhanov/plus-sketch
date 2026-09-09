/* host.c -- macOS/Linux side of plus-sketch.
 *
 *   ./host models/plus_sketch_q4.psk                    info + self-check
 *   ./host models/plus_sketch_q4.psk tests/ref_q4.bin   logit comparison
 *   ./host models/plus_sketch_q4.psk -d 3               draw category 3 -> SVG
 *   ./host models/plus_sketch_q4.psk -d 3 -t 0.2 -n 4   sampled, 4 drawings
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sketch.h"

/* ---- reference file --------------------------------------------------- */

typedef struct {
    int n_steps, vocab;
    int32_t *toks;
    float   *logits;
} refdata;

static int32_t rd32(const uint8_t *p)
{
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static float rdf32(const uint8_t *p)
{
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
          ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return c.f;
}

static int ref_load(refdata *r, const char *path)
{
    FILE *f = fopen(path, "rb");
    long size; uint8_t *b; int i;
    if (!f) return -1;
    fseek(f, 0, SEEK_END); size = ftell(f); fseek(f, 0, SEEK_SET);
    b = (uint8_t *)malloc((size_t)size);
    if (!b || fread(b, 1, (size_t)size, f) != (size_t)size) {
        free(b); fclose(f); return -1;
    }
    fclose(f);
    if (memcmp(b, "PREF", 4) != 0) { free(b); return -1; }
    r->n_steps = rd32(b + 4);
    r->vocab   = rd32(b + 8);
    r->toks   = (int32_t *)malloc(sizeof(int32_t) * (size_t)r->n_steps);
    r->logits = (float *)malloc(sizeof(float) *
                                (size_t)r->n_steps * (size_t)r->vocab);
    for (i = 0; i < r->n_steps; i++) r->toks[i] = rd32(b + 12 + i * 4);
    {
        const uint8_t *p = b + 12 + r->n_steps * 4;
        long n = (long)r->n_steps * r->vocab, k;
        for (k = 0; k < n; k++) r->logits[k] = rdf32(p + k * 4);
    }
    free(b);
    return 0;
}

static int compare(const psk_model *m, psk_state *s, const refdata *r)
{
    int i, j, bad = 0, ties = 0;
    double worst_abs = 0, worst_rel = 0;

    printf("\ncomparing %d steps against the reference:\n", r->n_steps);
    printf("  %-4s %-5s %-19s %-19s %s\n",
           "pos", "tok", "C argmax", "ref argmax", "max|diff|");

    for (i = 0; i < r->n_steps; i++) {
        const fx_t *lg = psk_forward(m, s, r->toks[i], i);
        const float *rf = r->logits + (long)i * r->vocab;
        int cbest = 0, rbest = 0;
        double mx = 0, scale = 0;
        float second = -1e30f;

        for (j = 1; j < r->vocab; j++) {
            if (lg[j] > lg[cbest]) cbest = j;
            if (rf[j] > rf[rbest]) rbest = j;
        }
        for (j = 0; j < r->vocab; j++) {
            double c = (double)lg[j] / FX_ONE, d = fabs(c - rf[j]);
            if (d > mx) mx = d;
            if (fabs(rf[j]) > scale) scale = fabs(rf[j]);
            if (j != rbest && rf[j] > second) second = rf[j];
        }
        if (mx > worst_abs) worst_abs = mx;
        if (scale > 0 && mx / scale > worst_rel) worst_rel = mx / scale;

        {
            int mism = (cbest != rbest);
            int tie  = mism && (rf[rbest] - second <= 0.1f);
            printf("  %-4d %-5d %-4d (%8.3f)    %-4d (%8.3f)    %.4f%s\n",
                   i, r->toks[i], cbest, (double)lg[cbest] / FX_ONE,
                   rbest, (double)rf[rbest], mx,
                   !mism ? "" : (tie ? "   (near-tie, ignored)"
                                     : "   <-- MISMATCH"));
            if (mism) { if (tie) ties++; else bad++; }
        }
    }

    printf("\n  worst absolute logit error: %.5f\n", worst_abs);
    printf("  worst relative error:       %.3f%%\n", 100 * worst_rel);
    printf("  mismatches: %d  (plus %d near-ties, ignored)\n", bad, ties);
    printf("\n  %s\n", bad == 0 ? "PASS -- the C forward pass matches PyTorch."
                                : "FAIL -- something in the forward pass is wrong.");
    return bad;
}

/* ---- SVG -------------------------------------------------------------- */

static void write_svg(const sketch *sk, const char *path, int size)
{
    FILE *f = fopen(path, "w");
    int st, i;
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" "
               "height=\"%d\" viewBox=\"0 0 %d %d\">\n", size, size, size, size);
    fprintf(f, "<rect width=\"%d\" height=\"%d\" fill=\"white\"/>\n", size, size);
    for (st = 0; st < sk->n_strokes; st++) {
        int a = sk->start[st];
        int b = (st + 1 < sk->n_strokes) ? sk->start[st + 1] : sk->n_pts;
        if (b - a < 2) continue;
        fprintf(f, "<path d=\"");
        for (i = a; i < b; i++)
            fprintf(f, "%c%d,%d ", i == a ? 'M' : 'L', sk->x[i], sk->y[i]);
        fprintf(f, "\" stroke=\"black\" stroke-width=\"3\" fill=\"none\" "
                   "stroke-linecap=\"round\"/>\n");
    }
    fprintf(f, "</svg>\n");
    fclose(f);
}

static void self_check(void)
{
    printf("\nfixed-point self-check:\n");
    printf("  sqrt(2)    = %.5f (want 1.41421)\n",
           (double)fx_sqrt(2 * FX_ONE) / FX_ONE);
    printf("  exp(-2)    = %.6f (want 0.135335)\n",
           (double)fx_exp_neg(-2 * FX_ONE) / FX_ONE);
    printf("  sin(1)     = %.5f (want 0.84147)\n",
           (double)fx_sin(FX_ONE) / FX_ONE);
    printf("  sigmoid(2) = %.5f (want 0.88080)\n",
           (double)fx_sigmoid(2 * FX_ONE) / FX_ONE);
}

int main(int argc, char **argv)
{
    psk_model m; psk_state s;
    const char *err = NULL, *path = NULL, *ref = NULL;
    int cat = -1, n = 1, i, rc = 0, size = 512;
    double temp = 0.0;
    uint32_t rng = 1337;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-d") && i + 1 < argc)      cat  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) rng  = (uint32_t)atoi(argv[++i]);
        else if (!path) path = argv[i];
        else ref = argv[i];
    }
    if (!path) path = "models/plus_sketch_q4.psk";

    if (psk_load(&m, path, &err) != 0) {
        fprintf(stderr, "load failed: %s\n", err ? err : "unknown");
        return 1;
    }
    printf("loaded %s\n", path);
    psk_print_config(&m);

    if (psk_state_init(&s, &m.cfg) != 0) {
        fprintf(stderr, "state init failed\n"); psk_free(&m); return 1;
    }
    printf("\nruntime state: %.1f KB\n", psk_state_bytes(&m.cfg) / 1024.0);

    if (cat >= 0) {
        fx_t inv = (temp > 0) ? (fx_t)(FX_ONE / temp) : 0;
        sketch sk;
        char out[128];
        printf("\ndrawing category %d, temperature %.2f%s\n",
               cat, temp, temp > 0 ? "" : " (greedy)");
        for (i = 0; i < n; i++) {
            memset(s.kcache, 0,
                   sizeof(fx_t) * (size_t)m.cfg.n_layers *
                   m.cfg.max_seq_len * m.cfg.kv_dim);
            memset(s.vcache, 0,
                   sizeof(fx_t) * (size_t)m.cfg.n_layers *
                   m.cfg.max_seq_len * m.cfg.kv_dim);
            sketch_generate(&m, &s, &sk, cat, inv, &rng, NULL, NULL);
            sketch_fit(&sk, size, 24);
            snprintf(out, sizeof(out), "c_cat%d_t%.2f_%d.svg", cat, temp, i);
            write_svg(&sk, out, size);
            printf("  %d: %d tokens, %d strokes, %d points, %s -> %s\n",
                   i, sk.n_tokens, sk.n_strokes, sk.n_pts,
                   sk.hit_eos ? "EOS" : "hit seq limit", out);
        }
    } else {
        self_check();
        if (ref) {
            refdata r;
            if (ref_load(&r, ref) != 0) {
                fprintf(stderr, "\ncannot read reference %s\n", ref); rc = 1;
            } else if (r.vocab != m.cfg.vocab_size) {
                fprintf(stderr, "\nvocab mismatch\n"); rc = 1;
            } else {
                rc = compare(&m, &s, &r) ? 1 : 0;
                free(r.toks); free(r.logits);
            }
        } else {
            printf("\n(-d CAT to draw, or pass a reference file to check)\n");
        }
    }

    psk_state_free(&s);
    psk_free(&m);
    return rc;
}
