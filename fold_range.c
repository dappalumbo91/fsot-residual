/* Adaptive range coder. Counts start at 1 and step by 1.
   The frequency table is not stored. Division stays in 64-bit limbs. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

typedef struct {
    uint64_t hi;
    uint64_t lo;
} U128;

static U128 mul_u64(uint64_t a, uint64_t b) {
    uint64_t a0 = (uint32_t)a;
    uint64_t a1 = a >> 32;
    uint64_t b0 = (uint32_t)b;
    uint64_t b1 = b >> 32;
    uint64_t p0 = a0 * b0;
    uint64_t p1 = a0 * b1;
    uint64_t p2 = a1 * b0;
    uint64_t p3 = a1 * b1;
    uint64_t mid = (p0 >> 32) + (uint32_t)p1 + (uint32_t)p2;
    U128 r;
    r.lo = (uint32_t)p0 | (mid << 32);
    r.hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    return r;
}

static U128 div_u128_u32(U128 n, uint32_t d) {
    uint32_t limb[4] = {
        (uint32_t)n.lo,
        (uint32_t)(n.lo >> 32),
        (uint32_t)n.hi,
        (uint32_t)(n.hi >> 32),
    };
    uint32_t q[4];
    uint32_t rem = 0;
    for (int i = 3; i >= 0; i--) {
        uint64_t cur = ((uint64_t)rem << 32) | limb[i];
        q[i] = (uint32_t)(cur / d);
        rem = (uint32_t)(cur % d);
    }
    U128 out;
    out.lo = (uint64_t)q[0] | ((uint64_t)q[1] << 32);
    out.hi = (uint64_t)q[2] | ((uint64_t)q[3] << 32);
    return out;
}

/* (span * num) / den, span = 2^64 * full + lo. Quotient fits our interval. */
static uint64_t span_mul_div(uint64_t span_lo, int span_hi, uint64_t num, uint32_t den) {
    U128 prod = mul_u64(span_lo, num);
    if (span_hi) prod.hi += num;
    U128 q = div_u128_u32(prod, den);
    return q.lo;
}

typedef struct {
    uint32_t *fw;
    uint32_t *cnt;
    uint32_t uniq;
    uint32_t total;
} Model;

static int read_full(void *dst, size_t n) {
    return fread(dst, 1, n, stdin) == n;
}

static int write_full(const void *src, size_t n) {
    return fwrite(src, 1, n, stdout) == n;
}

static void fw_add(Model *m, uint32_t i, uint32_t d) {
    for (uint32_t x = i + 1u; x <= m->uniq; x += x & (0u - x)) m->fw[x] += d;
}

static uint32_t fw_prefix(const Model *m, uint32_t i) {
    uint32_t s = 0;
    for (uint32_t x = i; x != 0u; x &= x - 1u) s += m->fw[x];
    return s;
}

static uint32_t fw_find(const Model *m, uint32_t offset) {
    uint32_t idx = 0;
    uint32_t bit = 1u;
    while ((bit << 1) <= m->uniq) bit <<= 1;
    uint32_t remain = offset;
    for (; bit != 0u; bit >>= 1) {
        uint32_t next = idx + bit;
        if (next <= m->uniq && m->fw[next] <= remain) {
            idx = next;
            remain -= m->fw[next];
        }
    }
    return idx;
}

static int model_init(Model *m, uint32_t uniq) {
    m->uniq = uniq;
    m->total = uniq;
    m->cnt = (uint32_t *)calloc(uniq ? uniq : 1u, sizeof(uint32_t));
    m->fw = (uint32_t *)calloc((size_t)uniq + 1u, sizeof(uint32_t));
    if (!m->cnt || !m->fw) return 0;
    for (uint32_t i = 0; i < uniq; i++) {
        m->cnt[i] = 1u;
        fw_add(m, i, 1u);
    }
    return 1;
}

static void model_free(Model *m) {
    free(m->cnt);
    free(m->fw);
}

typedef struct {
    uint8_t *buf;
    size_t n;
    size_t cap;
    uint32_t acc;
    int acc_n;
    int pending;
    uint64_t low;
    uint64_t high;
} Enc;

static int enc_grow(Enc *e, size_t need) {
    if (e->n + need <= e->cap) return 1;
    size_t cap = e->cap ? e->cap : 256u;
    while (e->n + need > cap) cap *= 2u;
    uint8_t *next = (uint8_t *)realloc(e->buf, cap);
    if (!next) return 0;
    e->buf = next;
    e->cap = cap;
    return 1;
}

static int put_bit(Enc *e, int bit) {
    e->acc = (e->acc << 1) | (uint32_t)(bit & 1);
    e->acc_n++;
    if (e->acc_n == 8) {
        if (!enc_grow(e, 1)) return 0;
        e->buf[e->n++] = (uint8_t)e->acc;
        e->acc = 0;
        e->acc_n = 0;
    }
    return 1;
}

static int emit_bit(Enc *e, int bit) {
    if (!put_bit(e, bit)) return 0;
    while (e->pending > 0) {
        if (!put_bit(e, bit ^ 1)) return 0;
        e->pending--;
    }
    return 1;
}

static int enc_scale(Enc *e) {
    for (;;) {
        if (e->high < 0x8000000000000000ull) {
            if (!emit_bit(e, 0)) return 0;
        } else if (e->low >= 0x8000000000000000ull) {
            if (!emit_bit(e, 1)) return 0;
            e->low -= 0x8000000000000000ull;
            e->high -= 0x8000000000000000ull;
        } else if (e->low >= 0x4000000000000000ull && e->high < 0xC000000000000000ull) {
            e->pending++;
            e->low -= 0x4000000000000000ull;
            e->high -= 0x4000000000000000ull;
        } else {
            break;
        }
        e->low <<= 1;
        e->high = (e->high << 1) | 1ull;
    }
    return 1;
}

static void span_of(uint64_t low, uint64_t high, uint64_t *span_lo, int *span_hi) {
    if (low == 0ull && high == 0xFFFFFFFFFFFFFFFFull) {
        *span_lo = 0;
        *span_hi = 1;
        return;
    }
    *span_lo = high - low + 1ull;
    *span_hi = 0;
}

static int enc_symbol(Enc *e, uint32_t cum, uint32_t freq, uint32_t total) {
    if (freq == 0u || cum + freq > total || e->high < e->low) return 0;
    uint64_t span_lo;
    int span_hi;
    span_of(e->low, e->high, &span_lo, &span_hi);
    uint64_t nl = e->low + span_mul_div(span_lo, span_hi, cum, total);
    uint64_t nh;
    if (span_hi && (uint64_t)cum + freq == (uint64_t)total) {
        nh = e->high;
    } else {
        uint64_t end = span_mul_div(span_lo, span_hi, (uint64_t)cum + freq, total);
        if (end == 0ull) return 0;
        nh = e->low + end - 1ull;
    }
    if (nh < nl) return 0;
    e->low = nl;
    e->high = nh;
    return enc_scale(e);
}

static int enc_finish(Enc *e) {
    e->pending++;
    if (e->low < 0x4000000000000000ull) {
        if (!emit_bit(e, 0)) return 0;
    } else if (!emit_bit(e, 1)) {
        return 0;
    }
    if (e->acc_n) {
        if (!enc_grow(e, 1)) return 0;
        e->buf[e->n++] = (uint8_t)(e->acc << (8 - e->acc_n));
    }
    return 1;
}

typedef struct {
    const uint8_t *buf;
    size_t n;
    size_t i;
    uint32_t acc;
    int acc_n;
    uint64_t low;
    uint64_t high;
    uint64_t code;
} Dec;

static int dec_bit(Dec *d) {
    if (d->acc_n == 0) {
        uint8_t byte = 0;
        if (d->i < d->n) byte = d->buf[d->i++];
        d->acc = byte;
        d->acc_n = 8;
    }
    d->acc_n--;
    return (int)((d->acc >> d->acc_n) & 1u);
}

static void dec_scale(Dec *d) {
    for (;;) {
        if (d->high < 0x8000000000000000ull) {
        } else if (d->low >= 0x8000000000000000ull) {
            d->code -= 0x8000000000000000ull;
            d->low -= 0x8000000000000000ull;
            d->high -= 0x8000000000000000ull;
        } else if (d->low >= 0x4000000000000000ull && d->high < 0xC000000000000000ull) {
            d->code -= 0x4000000000000000ull;
            d->low -= 0x4000000000000000ull;
            d->high -= 0x4000000000000000ull;
        } else {
            break;
        }
        d->low <<= 1;
        d->high = (d->high << 1) | 1ull;
        d->code = (d->code << 1) | (uint64_t)dec_bit(d);
    }
}

static void dec_init(Dec *d) {
    d->low = 0;
    d->high = 0xFFFFFFFFFFFFFFFFull;
    d->code = 0;
    for (int i = 0; i < 64; i++) d->code = (d->code << 1) | (uint64_t)dec_bit(d);
}

/* (hi:lo) / den. hi < den, so the quotient fits in 64 bits. */
static uint64_t div_u128_u64(uint64_t hi, uint64_t lo, uint64_t den) {
    if (den == 0ull) return 0ull;
    if (hi == 0ull) return lo / den;
    if ((den >> 32) == 0ull) {
        U128 n;
        n.hi = hi;
        n.lo = lo;
        return div_u128_u32(n, (uint32_t)den).lo;
    }
    if (hi >= den) return ~0ull;
    uint64_t q = 0ull;
    uint64_t r = hi;
    for (int i = 63; i >= 0; i--) {
        uint64_t top = r >> 63;
        r = (r << 1) | ((lo >> (unsigned)i) & 1ull);
        if (top || r >= den) {
            r -= den;
            q |= 1ull << (unsigned)i;
        }
    }
    return q;
}

static int div_exact(uint64_t hi, uint64_t lo, uint64_t den, uint64_t q) {
    if (hi >= den) return 0;
    U128 prod = mul_u64(q, den);
    if (prod.hi > hi || (prod.hi == hi && prod.lo > lo)) return 0;
    uint64_t r_lo = lo - prod.lo;
    uint64_t r_hi = hi - prod.hi - (lo < prod.lo ? 1ull : 0ull);
    if (r_hi != 0ull) return 0;
    return r_lo < den;
}

/* floor(((code - low + 1) * total - 1) / span). Matches the encoder floors. */
static uint32_t range_offset(uint64_t low, uint64_t high, uint64_t code, uint32_t total) {
    if (total <= 1u) return 0u;
    uint64_t moved = code - low;
    if (low == 0ull && high == 0xFFFFFFFFFFFFFFFFull) {
        if (moved == 0xFFFFFFFFFFFFFFFFull) return total - 1u;
        U128 num = mul_u64(moved + 1ull, (uint64_t)total);
        if (num.lo == 0ull) {
            num.hi--;
            num.lo = 0xFFFFFFFFFFFFFFFFull;
        } else {
            num.lo--;
        }
        return (uint32_t)num.hi;
    }
    uint64_t span = high - low + 1ull;
    U128 num = mul_u64(moved + 1ull, (uint64_t)total);
    if (num.lo == 0ull) {
        num.hi--;
        num.lo = 0xFFFFFFFFFFFFFFFFull;
    } else {
        num.lo--;
    }
    uint64_t q = div_u128_u64(num.hi, num.lo, span);
    if (q >= total) return total - 1u;
    return (uint32_t)q;
}

static void narrow(uint64_t *low, uint64_t *high, uint32_t cum, uint32_t freq, uint32_t total) {
    uint64_t span_lo;
    int span_hi;
    span_of(*low, *high, &span_lo, &span_hi);
    uint64_t nl = *low + span_mul_div(span_lo, span_hi, cum, total);
    uint64_t nh;
    if (span_hi && (uint64_t)cum + freq == (uint64_t)total) nh = *high;
    else nh = *low + span_mul_div(span_lo, span_hi, (uint64_t)cum + freq, total) - 1ull;
    *low = nl;
    *high = nh;
}

static int roundtrip(const uint16_t *idx, uint32_t n, uint32_t uniq) {
    Model m;
    if (!model_init(&m, uniq)) return 4;
    Enc e;
    memset(&e, 0, sizeof(e));
    e.low = 0;
    e.high = 0xFFFFFFFFFFFFFFFFull;
    for (uint32_t i = 0; i < n; i++) {
        uint16_t sym = idx[i];
        if (sym >= uniq) {
            model_free(&m);
            free(e.buf);
            return 8;
        }
        uint32_t cum = fw_prefix(&m, sym);
        if (!enc_symbol(&e, cum, m.cnt[sym], m.total)) {
            model_free(&m);
            free(e.buf);
            return 9;
        }
        m.cnt[sym] += 1u;
        fw_add(&m, sym, 1u);
        m.total += 1u;
    }
    if (!enc_finish(&e)) {
        model_free(&m);
        free(e.buf);
        return 10;
    }
    model_free(&m);
    if (!model_init(&m, uniq)) {
        free(e.buf);
        return 4;
    }
    Dec d;
    memset(&d, 0, sizeof(d));
    d.buf = e.buf;
    d.n = e.n;
    dec_init(&d);
    for (uint32_t i = 0; i < n; i++) {
        if (d.high < d.low || d.code < d.low || d.code > d.high) {
            model_free(&m);
            free(e.buf);
            return 11;
        }
        uint32_t offset = range_offset(d.low, d.high, d.code, m.total);
        uint32_t sym = fw_find(&m, offset);
        uint32_t cum = fw_prefix(&m, sym);
        uint32_t freq = m.cnt[sym];
        if (sym >= uniq || offset < cum || offset >= cum + freq) {
            model_free(&m);
            free(e.buf);
            return 12;
        }
        narrow(&d.low, &d.high, cum, freq, m.total);
        dec_scale(&d);
        if (sym != idx[i]) {
            model_free(&m);
            free(e.buf);
            return 14;
        }
        m.cnt[sym] += 1u;
        fw_add(&m, sym, 1u);
        m.total += 1u;
    }
    model_free(&m);
    free(e.buf);
    return 0;
}

static int self_main(void) {
    if (div_u128_u64(0ull, 100ull, 7ull) != 14ull) return 70;
    if (!div_exact(0ull, 100ull, 7ull, 14ull)) return 71;
    if (div_u128_u64(1ull, 5ull, 3ull) != 6148914691236517207ull) return 72;
    if (!div_exact(1ull, 5ull, 3ull, 6148914691236517207ull)) return 73;
    uint64_t div_state = 1ull;
    for (int i = 0; i < 2000; i++) {
        div_state = div_state * 1664525ull + 1013904223ull;
        uint64_t den = (div_state | 1ull);
        div_state = div_state * 1664525ull + 1013904223ull;
        uint64_t hi = div_state % den;
        div_state = div_state * 1664525ull + 1013904223ull;
        uint64_t lo = div_state;
        uint64_t q = div_u128_u64(hi, lo, den);
        if (!div_exact(hi, lo, den, q)) return 74;
    }
    uint16_t bseq[] = {1, 0, 2, 2, 0, 1, 3, 0};
    int rc = roundtrip(bseq, 8, 4);
    if (rc != 0) return 80 + rc;
    uint16_t a[] = {0, 1, 2, 1, 0, 0, 2, 2, 1};
    rc = roundtrip(a, 9, 3);
    if (rc != 0) return 20 + rc;
    uint16_t one[64];
    for (int i = 0; i < 64; i++) one[i] = 0;
    rc = roundtrip(one, 64, 1);
    if (rc != 0) return 40 + rc;
    uint16_t rnd[4000];
    uint32_t state = 1u;
    for (int i = 0; i < 4000; i++) {
        state = state * 1664525u + 1013904223u;
        rnd[i] = (uint16_t)(state % 17u);
    }
    rc = roundtrip(rnd, 4000, 17);
    if (rc != 0) return 60 + rc;
    return 0;
}

static int encode_main(void) {
    uint32_t n = 0, uniq = 0;
    if (!read_full(&n, 4) || !read_full(&uniq, 4)) return 2;
    if (uniq == 0u || uniq > 65535u) return 3;
    uint16_t *idx = (uint16_t *)calloc(n ? n : 1u, sizeof(uint16_t));
    if (!idx) return 4;
    if (n && !read_full(idx, (size_t)n * 2u)) {
        free(idx);
        return 5;
    }
    Model m;
    if (!model_init(&m, uniq)) {
        free(idx);
        return 4;
    }
    Enc e;
    memset(&e, 0, sizeof(e));
    e.low = 0;
    e.high = 0xFFFFFFFFFFFFFFFFull;
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < n; i++) {
        uint16_t sym = idx[i];
        if (sym >= uniq) rc = 8;
        else {
            uint32_t cum = fw_prefix(&m, sym);
            if (!enc_symbol(&e, cum, m.cnt[sym], m.total)) rc = 9;
            else {
                m.cnt[sym] += 1u;
                fw_add(&m, sym, 1u);
                m.total += 1u;
            }
        }
    }
    if (rc == 0 && !enc_finish(&e)) rc = 10;
    if (rc == 0) {
        uint32_t len = (uint32_t)e.n;
        if (!write_full(&len, 4) || (e.n && !write_full(e.buf, e.n))) rc = 10;
    }
    model_free(&m);
    free(e.buf);
    free(idx);
    return rc;
}

static int decode_main(void) {
    uint32_t n = 0, uniq = 0, payload_len = 0;
    if (!read_full(&n, 4) || !read_full(&uniq, 4) || !read_full(&payload_len, 4)) return 2;
    if (uniq == 0u || uniq > 65535u) return 3;
    uint8_t *payload = (uint8_t *)malloc(payload_len ? payload_len : 1u);
    uint16_t *idx = (uint16_t *)calloc(n ? n : 1u, sizeof(uint16_t));
    if (!payload || !idx) return 4;
    if (payload_len && !read_full(payload, payload_len)) return 6;
    Model m;
    if (!model_init(&m, uniq)) return 4;
    Dec d;
    memset(&d, 0, sizeof(d));
    d.buf = payload;
    d.n = payload_len;
    dec_init(&d);
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < n; i++) {
        if (d.high < d.low || d.code < d.low || d.code > d.high) rc = 11;
        else {
            uint32_t offset = range_offset(d.low, d.high, d.code, m.total);
            uint32_t sym = fw_find(&m, offset);
            uint32_t cum = fw_prefix(&m, sym);
            uint32_t freq = m.cnt[sym];
            if (sym >= uniq || offset < cum || offset >= cum + freq) rc = 12;
            else {
                narrow(&d.low, &d.high, cum, freq, m.total);
                dec_scale(&d);
                idx[i] = (uint16_t)sym;
                m.cnt[sym] += 1u;
                fw_add(&m, sym, 1u);
                m.total += 1u;
            }
        }
    }
    if (rc == 0 && n && !write_full(idx, (size_t)n * 2u)) rc = 12;
    model_free(&m);
    free(payload);
    free(idx);
    return rc;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc != 2) return 1;
    if (strcmp(argv[1], "self") == 0) return self_main();
    if (strcmp(argv[1], "enc") == 0) return encode_main();
    if (strcmp(argv[1], "dec") == 0) return decode_main();
    return 1;
}
