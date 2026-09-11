/* mac68k.c -- plus-sketch on a Macintosh Plus.
 *
 * QuickDraw from the 128K ROM. No RetroConsole (C++), no stdio, and no
 * floating point anywhere: one %f in printf drags in newlib's FP formatting,
 * which on a machine with no FPU means SANE and ~1.2 MB of code. core/ has
 * no floats, so the platform layer must not reintroduce them.
 *
 * The model is read with the File Manager, so core/psk.c is built with
 * -DPSK_NO_STDIO.
 *
 * Drawing streams: sketch_generate() calls on_point() after every point, so
 * the picture appears stroke by stroke over the several minutes it takes,
 * rather than all at once at the end. The bounding box is tracked as it
 * grows and the canvas is rescaled and redrawn when a point falls outside
 * it -- with hysteresis, so a slowly expanding drawing doesn't redraw on
 * every single point.
 */

#include <Quickdraw.h>
#include <Windows.h>
#include <Fonts.h>
#include <Events.h>
#include <Menus.h>
#include <TextEdit.h>
#include <Dialogs.h>
#include <Files.h>
#include <MacMemory.h>

#include <stdlib.h>
#include <string.h>

#include "sketch.h"

#define MODEL_NAME "plus_sketch_q4.psk"
#define CATS_NAME  "categories.txt"

/* window: 512x342 screen, menu bar is 20 */
#define WIN_L    2
#define WIN_T    24
#define WIN_R   510
#define WIN_B   340

#define PROMPT_Y  14        /* baselines, window-relative */
#define TEMP_Y    28
#define CANVAS_T  34
#define CANVAS_B  282
#define STATUS_Y  296
#define INFO_Y    310

static WindowPtr gWin;
static Rect      gCanvas;

/* ---- tiny text helpers (no stdio) ------------------------------------- */

static char gL[128];
static int  gP;

static void put_s(const char *s)
{
    while (*s && gP < (int)sizeof(gL) - 1) gL[gP++] = *s++;
}

static void put_n(long v)
{
    char t[16];
    int n = 0, neg = 0;
    if (v < 0) { neg = 1; v = -v; }
    if (v == 0) t[n++] = '0';
    while (v > 0 && n < 15) { t[n++] = (char)('0' + (v % 10)); v /= 10; }
    if (neg && gP < (int)sizeof(gL) - 1) gL[gP++] = '-';
    while (n > 0 && gP < (int)sizeof(gL) - 1) gL[gP++] = t[--n];
}

/* num/den to two decimals, integers only */
static void put_frac(long num, long den)
{
    long w, h;
    if (den == 0) { put_s("--"); return; }
    w = num / den;
    h = ((num - w * den) * 100) / den;
    if (h < 0) h = -h;
    put_n(w); put_s(".");
    if (h < 10) put_s("0");
    put_n(h);
}

/* Draw the accumulated line at a fixed baseline, clearing that row first. */
static void flush_at(int y)
{
    Rect r;
    SetRect(&r, WIN_L + 2, y - 10, WIN_R - WIN_L - 4, y + 3);
    EraseRect(&r);
    MoveTo(6, y);
    DrawText(gL, 0, gP);
    gP = 0;
}

/* ---- file loading ----------------------------------------------------- */

static OSErr read_file(const char *name, void **data, long *size)
{
    Str255 pn;
    short ref;
    long len;
    OSErr err;
    char *p;
    size_t n = strlen(name);

    if (n > 255) n = 255;
    pn[0] = (unsigned char)n;
    memcpy(pn + 1, name, n);

    err = FSOpen(pn, 0, &ref);
    if (err) return err;
    err = GetEOF(ref, &len);
    if (err) { FSClose(ref); return err; }
    p = (char *)malloc((size_t)len + 1);
    if (!p) { FSClose(ref); return memFullErr; }
    err = FSRead(ref, &len, p);
    FSClose(ref);
    if (err && err != eofErr) { free(p); return err; }
    p[len] = '\0';
    *data = p;
    *size = len;
    return noErr;
}

/* ---- categories ------------------------------------------------------- */

#define MAX_CATS 512
static char  *gCatBuf;
static char  *gCat[MAX_CATS];
static int    gNCats;

static int load_categories(void)
{
    long size;
    void *blob;
    char *p, *end;
    if (read_file(CATS_NAME, &blob, &size) != noErr) return -1;
    gCatBuf = (char *)blob;
    p = gCatBuf;
    end = gCatBuf + size;
    gNCats = 0;
    while (p < end && gNCats < MAX_CATS) {
        char *line = p;
        while (p < end && *p != '\n' && *p != '\r') p++;
        *p = '\0';
        p++;
        while (p < end && (*p == '\n' || *p == '\r')) p++;
        if (*line) gCat[gNCats++] = line;
    }
    return 0;
}

static char lower(char c) { return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c; }

static int same(const char *a, const char *b)
{
    while (*a && *b) {
        if (lower(*a) != lower(*b)) return 0;
        a++; b++;
    }
    return *a == *b;
}

/* exact, then plural strip. -1 if unknown. */
static int find_category(const char *w)
{
    int i;
    char tmp[64];
    size_t n = strlen(w);
    for (i = 0; i < gNCats; i++)
        if (same(w, gCat[i])) return i;
    if (n > 1 && n < 63 && lower(w[n - 1]) == 's') {
        memcpy(tmp, w, n - 1);
        tmp[n - 1] = '\0';
        for (i = 0; i < gNCats; i++)
            if (same(tmp, gCat[i])) return i;
    }
    return -1;
}

/* ---- streaming canvas ------------------------------------------------- */

static int32_t bx0, by0, bx1, by1;      /* bounding box, model coords */

static void box_reset(void)
{
    bx0 = -40; by0 = -40; bx1 = 40; by1 = 40;
}

/* Grow the box to include (x,y), with padding so a steadily expanding
 * drawing doesn't force a redraw on every point. */
static void box_grow(int32_t x, int32_t y)
{
    int32_t pad;
    if (x < bx0) bx0 = x;
    if (x > bx1) bx1 = x;
    if (y < by0) by0 = y;
    if (y > by1) by1 = y;
    pad = ((bx1 - bx0) + (by1 - by0)) / 16;
    if (pad < 8) pad = 8;
    bx0 -= pad; bx1 += pad; by0 -= pad; by1 += pad;
}

/* Model coords -> screen. Integer only; the products stay well inside
 * int32 for any plausible sketch. */
static void map_pt(int32_t mx, int32_t my, int *sx, int *sy)
{
    int32_t bw = bx1 - bx0, bh = by1 - by0;
    int32_t cw = gCanvas.right - gCanvas.left;
    int32_t ch = gCanvas.bottom - gCanvas.top;
    int32_t sxn, syn, span, cmin;

    if (bw < 1) bw = 1;
    if (bh < 1) bh = 1;
    /* uniform scale: fit the larger extent */
    span = (bw > bh) ? bw : bh;
    cmin = (cw < ch) ? cw : ch;

    sxn = ((mx - bx0) * cmin) / span;
    syn = ((my - by0) * cmin) / span;
    *sx = (int)(gCanvas.left + (cw - cmin) / 2 + sxn);
    *sy = (int)(gCanvas.top  + (ch - cmin) / 2 + syn);
}

static void redraw_all(const sketch *sk)
{
    int st, i, sx, sy;
    EraseRect(&gCanvas);
    FrameRect(&gCanvas);
    for (st = 0; st < sk->n_strokes; st++) {
        int a = sk->start[st];
        int b = (st + 1 < sk->n_strokes) ? sk->start[st + 1] : sk->n_pts;
        if (b - a < 2) continue;
        map_pt(sk->x[a], sk->y[a], &sx, &sy);
        MoveTo(sx, sy);
        for (i = a + 1; i < b; i++) {
            map_pt(sk->x[i], sk->y[i], &sx, &sy);
            LineTo(sx, sy);
        }
    }
}

static int gTokCount;

static void on_point(void *ctx, int32_t x, int32_t y, int pen_down)
{
    sketch *sk = (sketch *)ctx;
    int sx, sy, px, py;

    gTokCount++;
    gP = 0;
    put_s("drawing... "); put_n(gTokCount); put_s(" points");
    flush_at(STATUS_Y);

    if (x < bx0 || x > bx1 || y < by0 || y > by1) {
        box_grow(x, y);
        redraw_all(sk);
        return;
    }
    if (pen_down && sk->n_pts >= 2) {
        map_pt(sk->x[sk->n_pts - 2], sk->y[sk->n_pts - 2], &px, &py);
        map_pt(x, y, &sx, &sy);
        MoveTo(px, py);
        LineTo(sx, sy);
    }
}

/* ---- prompt ----------------------------------------------------------- */

static char gWord[40];
static int  gWordLen;
static int  gTempH = 20;        /* temperature x100, 0..60 */

static void draw_prompt(int caret)
{
    gP = 0;
    put_s("DRAW A ");
    if (gWordLen == 0) put_s("____");
    else { gWord[gWordLen] = '\0'; put_s(gWord); }
    if (caret) put_s("_");
    flush_at(PROMPT_Y);

    gP = 0;
    put_s("TEMP ");
    if (gTempH == 0) put_s("0.00 (greedy)");
    else put_frac(gTempH, 100);
    put_s("   left/right to change, return to draw");
    flush_at(TEMP_Y);
}

static fx_t temp_inv(void)
{
    if (gTempH <= 0) return 0;
    return (fx_t)((65536L * 100L) / gTempH);
}

/* Blocking read of one keypress. */
static int get_key(void)
{
    EventRecord ev;
    for (;;) {
        if (GetNextEvent(keyDownMask | autoKeyMask, &ev))
            return (int)(ev.message & charCodeMask);
    }
}

/* ---- main ------------------------------------------------------------- */

int main(void)
{
    psk_model m;
    psk_state s;
    sketch sk;
    const char *err = NULL;
    void *blob = NULL;
    long blobsz = 0;
    uint32_t rng = 1337;
    OSErr oe;
    Rect r;

    InitGraf(&qd.thePort);
    InitFonts();
    InitWindows();
    InitMenus();
    TEInit();
    InitDialogs(NULL);
    InitCursor();

    SetRect(&r, WIN_L, WIN_T, WIN_R, WIN_B);
    gWin = NewWindow(NULL, &r, "\pplus-sketch", true,
                     documentProc, (WindowPtr)-1L, false, 0);
    SetPort(gWin);
    TextFont(4);            /* Monaco */
    TextSize(9);

    SetRect(&gCanvas, 6, CANVAS_T, WIN_R - WIN_L - 8, CANVAS_B);

    gP = 0; put_s("loading model..."); flush_at(PROMPT_Y);

    oe = read_file(MODEL_NAME, &blob, &blobsz);
    if (oe != noErr) {
        gP = 0; put_s("cannot read " MODEL_NAME ", OSErr "); put_n((long)oe);
        flush_at(PROMPT_Y);
        get_key();
        return 1;
    }
    if (psk_load_mem(&m, blob, blobsz, &err) != 0) {
        gP = 0; put_s("bad model: "); put_s(err ? err : "?");
        flush_at(PROMPT_Y);
        get_key();
        return 1;
    }
    if (load_categories() != 0) {
        gP = 0; put_s("cannot read " CATS_NAME);
        flush_at(PROMPT_Y);
        get_key();
        return 1;
    }
    if (psk_state_init(&s, &m.cfg) != 0) {
        gP = 0; put_s("out of memory (state needs ");
        put_n(psk_state_bytes(&m.cfg) / 1024); put_s(" K)");
        flush_at(PROMPT_Y);
        psk_free(&m);
        get_key();
        return 1;
    }

    gP = 0;
    put_n(gNCats); put_s(" things I know how to draw");
    flush_at(INFO_Y);

    EraseRect(&gCanvas);
    FrameRect(&gCanvas);

    for (;;) {
        int c, cat;
        unsigned long t0, t1;

        gWordLen = 0;
        draw_prompt(1);

        /* read a word */
        for (;;) {
            c = get_key();
            if (c == '\r' || c == 3) break;              /* return / enter */
            if (c == 8 && gWordLen > 0) { gWordLen--; }  /* backspace */
            else if (c == 0x1C && gTempH > 0)  gTempH -= 5;   /* left */
            else if (c == 0x1D && gTempH < 60) gTempH += 5;   /* right */
            else if (c == 0x1B) { gWordLen = 0; }             /* esc */
            else if (c >= 32 && c < 127 && gWordLen < 38)
                gWord[gWordLen++] = (char)c;
            draw_prompt(1);
        }
        if (gWordLen == 0) continue;
        gWord[gWordLen] = '\0';

        cat = find_category(gWord);
        if (cat < 0) {
            gP = 0;
            put_s("I DON'T KNOW HOW TO DRAW A ");
            put_s(gWord);
            put_s(".");
            flush_at(STATUS_Y);
            continue;
        }

        gP = 0; put_s("drawing "); put_s(gCat[cat]); put_s("...");
        flush_at(STATUS_Y);
        gP = 0; flush_at(INFO_Y);

        memset(s.kcache, 0, sizeof(fx_t) * (size_t)m.cfg.n_layers *
               m.cfg.max_seq_len * m.cfg.kv_dim);
        memset(s.vcache, 0, sizeof(fx_t) * (size_t)m.cfg.n_layers *
               m.cfg.max_seq_len * m.cfg.kv_dim);

        box_reset();
        EraseRect(&gCanvas);
        FrameRect(&gCanvas);
        gTokCount = 0;

        t0 = TickCount();
        sketch_generate(&m, &s, &sk, cat, temp_inv(), &rng, on_point, &sk);
        t1 = TickCount();

        redraw_all(&sk);

        gP = 0;
        put_s(gCat[cat]); put_s(" -- ");
        put_n(sk.n_tokens); put_s(" tokens, ");
        put_n(sk.n_strokes); put_s(" strokes");
        flush_at(STATUS_Y);

        gP = 0;
        put_frac((long)(t1 - t0), 60); put_s(" s total, ");
        put_frac((long)(t1 - t0), sk.n_tokens ? sk.n_tokens * 60L : 60L);
        put_s(" s/token   ");
        put_s(sk.hit_eos ? "(EOS)" : "(hit limit)");
        flush_at(INFO_Y);
    }
}