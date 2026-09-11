/* fixed.h -- 16.16 fixed-point arithmetic.
 *
 * The 68000 has no FPU. Every float operation goes through SANE at hundreds
 * of cycles, so there is no floating point anywhere in core/.
 *
 * Format: int32_t holding value * 65536. Range +/-32767.99998,
 * resolution 1/65536 = 1.5e-5.
 *
 * Where fixed point is used:  scales, RMSNorm, softmax, RoPE, temperature.
 * Where it is NOT used:  the matmul inner loop, which is int8 x int4 summed
 * into an int32 accumulator -- plain integer arithmetic, no shifts per
 * element. Scales are applied once per group of 64. See docs.
 */

#ifndef PSK_FIXED_H
#define PSK_FIXED_H

#include <stdint.h>

typedef int32_t fx_t;

#define FX_SHIFT 16
#define FX_ONE   (1L << FX_SHIFT)
#define FX_HALF  (FX_ONE / 2)

#define FX_FROM_INT(n)  ((fx_t)((n) << FX_SHIFT))
#define FX_TO_INT(a)    ((int32_t)((a) >> FX_SHIFT))

/* Rounded conversion to int, for final coordinates. */
static inline int32_t fx_round(fx_t a)
{
    return (int32_t)((a + (a >= 0 ? FX_HALF : -FX_HALF)) >> FX_SHIFT);
}

/* 16x16 -> 32 multiply. This is the ONLY multiply that is cheap on a
 * 68000: MULS.W, about 70 cycles. Anything wider becomes a __mulsi3 or
 * __muldi3 helper call at several hundred cycles.
 *
 * Writing it with int16_t operands is what lets GCC's m68k backend match
 * its mulhisi3 pattern. Passing plain ints -- even ints known to be small --
 * gets you the helper. This cost the matmul 665 cycles per MAC before it
 * was found.
 */
static inline int32_t mul16(int16_t a, int16_t b)
{
    return (int32_t)a * (int32_t)b;
}

/* Saturating convert 16.16 -> 8.8, for feeding mul16. */
static inline int16_t to_88(fx_t v)
{
    v >>= 8;
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

/* Multiply. Needs a 64-bit intermediate: two 16.16 values multiplied give
 * a 32.32 result which must be shifted back down.
 *
 * On the 68000 this compiles to a __mulsi3-style helper (no 32x32->64
 * instruction until the 68020's MULS.L). It is therefore expensive --
 * roughly 200+ cycles -- which is exactly why it must stay OUT of the
 * matmul inner loop and appear only once per group of 64 weights.
 */
static inline fx_t fx_mul(fx_t a, fx_t b)
{
    return (fx_t)(((int64_t)a * (int64_t)b) >> FX_SHIFT);
}

/* Divide. Even more expensive than fx_mul; avoid in hot paths.
 * Temperature is applied as a precomputed reciprocal instead. */
static inline fx_t fx_div(fx_t a, fx_t b)
{
    if (b == 0) return a >= 0 ? INT32_MAX : INT32_MIN;
    return (fx_t)(((int64_t)a << FX_SHIFT) / b);
}

/* Integer square root of a 32-bit value (Newton-free, bit-by-bit).
 * ~32 iterations of shift/compare/subtract -- no division, no multiply. */
static inline uint32_t isqrt32(uint32_t n)
{
    uint32_t rem = 0, root = 0, i;
    for (i = 0; i < 16; i++) {
        root <<= 1;
        rem = (rem << 2) | (n >> 30);
        n <<= 2;
        if (root < rem) {
            rem -= root | 1;
            root += 2;
        }
    }
    return root >> 1;
}

/* sqrt of a 16.16 value, result in 16.16.
 *
 * We want sqrt(x) * 2^16 given a = x * 2^16.
 *   isqrt(a << 8) = sqrt(x) * 2^12, so shift up 4 more.
 * That overflows once a >= 2^24 (x >= 256), so above that use
 *   isqrt(a) = sqrt(x) * 2^8, shifted up 8.
 * Accuracy is ~1e-4 in the common range, which is well inside int4
 * quantization error. */
static inline fx_t fx_sqrt(fx_t a)
{
    if (a <= 0) return 0;
    if ((uint32_t)a < (1UL << 24))
        return (fx_t)(isqrt32((uint32_t)a << 8) << 4);
    return (fx_t)(isqrt32((uint32_t)a) << 8);
}

/* Reciprocal square root, 16.16. Used by RMSNorm. */
static inline fx_t fx_rsqrt(fx_t a)
{
    fx_t s = fx_sqrt(a);
    if (s == 0) return 0;
    return fx_div(FX_ONE, s);
}

/* Clamp helpers. */
static inline int32_t clamp32(int32_t v, int32_t lo, int32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}


/* ---- transcendentals, from tables.h ---------------------------------- */

#include "tables.h"

/* exp(x) for x <= 0, result in 16.16. Returns 0 for very negative x.
 * exp(x) = 2^(x*log2e); split into integer and fractional parts, take
 * 2^frac from the table and shift by the integer part. */
static inline fx_t fx_exp_neg(fx_t x)
{
    fx_t t, frac;
    int32_t ip, idx;
    if (x >= 0) return FX_ONE;
    t = fx_mul(x, FX_LOG2E);          /* t <= 0 */
    ip = t >> FX_SHIFT;               /* floor, since >> rounds down */
    if (ip < -30) return 0;
    frac = t - (ip << FX_SHIFT);      /* in [0, 1) */

    /* EXP2_N is 65, so the table step is 65536/64 = 1024 = 2^10. That makes
     * the interpolation shifts rather than a 64-bit multiply and divide --
     * this function is called ~2800 times per token. */
    idx = frac >> 10;
    if (idx > EXP2_N - 2) idx = EXP2_N - 2;
    {
        fx_t f0 = exp2_tab[idx], f1 = exp2_tab[idx + 1];
        int16_t d = (int16_t)(f1 - f0);          /* small, fits 16 bits */
        int16_t r = (int16_t)(frac & 1023);
        fx_t v = f0 + (mul16(d, r) >> 10);
        return v >> (-ip);
    }
}

/* Same as fx_exp_neg but returns 8.24.
 *
 * The final shift in fx_exp_neg is where small results lose their low bits:
 * exp(-6.3) is about 0.0056, which is only 368 units of 16.16. RoPE's
 * slowest frequency channels are exactly that size, and the error is then
 * multiplied by the position -- so by position 100 the sine is 3.5% wrong.
 * Keeping 8 more fractional bits fixes it. */
static inline int64_t fx_exp_neg24(fx_t x)
{
    fx_t t, frac;
    int32_t ip, idx;
    if (x >= 0) return (int64_t)FX_ONE << 8;
    t = fx_mul(x, FX_LOG2E);
    ip = t >> FX_SHIFT;
    if (ip < -30) return 0;
    frac = t - (ip << FX_SHIFT);
    idx = frac >> 10;
    if (idx > EXP2_N - 2) idx = EXP2_N - 2;
    {
        fx_t f0 = exp2_tab[idx], f1 = exp2_tab[idx + 1];
        int16_t d = (int16_t)(f1 - f0);
        int16_t r = (int16_t)(frac & 1023);
        fx_t v = f0 + (mul16(d, r) >> 10);
        return (((int64_t)v) << 8) >> (-ip);
    }
}

/* sigmoid(x) = 1 / (1 + exp(-x)), 16.16. */
static inline fx_t fx_sigmoid(fx_t x)
{
    if (x >= 0) {
        fx_t e = fx_exp_neg(-x);
        return fx_div(FX_ONE, FX_ONE + e);
    } else {
        fx_t e = fx_exp_neg(x);
        return fx_div(e, FX_ONE + e);
    }
}

/* sin/cos, 16.16, any input angle. Quadrant reduction onto [0, pi/2]. */
static fx_t fx_sin(fx_t x)
{
    int neg = 0;
    int32_t idx, step;
    fx_t rem, v0, v1;

    /* fold into [0, 2pi) */
    while (x < 0)         x += FX_TWO_PI;
    while (x >= FX_TWO_PI) x -= FX_TWO_PI;
    if (x >= FX_PI) { x -= FX_PI; neg = 1; }
    if (x > FX_HALF_PI) x = FX_PI - x;

    step = FX_HALF_PI / (SIN_N - 1);
    idx  = x / step;
    if (idx >= SIN_N - 1) return neg ? -FX_ONE : FX_ONE;
    rem = x - idx * step;
    v0 = sin_tab[idx]; v1 = sin_tab[idx + 1];
    {
        fx_t v = v0 + (fx_t)(((int64_t)(v1 - v0) * rem) / step);
        return neg ? -v : v;
    }
}

static inline fx_t fx_cos(fx_t x) { return fx_sin(x + FX_HALF_PI); }

#endif /* PSK_FIXED_H */
