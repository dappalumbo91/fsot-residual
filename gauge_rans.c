/* Static order-0 rANS for one declared alphabet. Same renorm as fspack.py. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define SCALE_BITS 12u
#define SCALE (1u << SCALE_BITS)
#define RANS_L (1u << 23)

static int read_full(void *dst, size_t n) {
    return fread(dst, 1, n, stdin) == n;
}

static int write_full(const void *src, size_t n) {
    return fwrite(src, 1, n, stdout) == n;
}

static int encode_main(void) {
    uint32_t n = 0, uniq = 0;
    if (!read_full(&n, 4) || !read_full(&uniq, 4)) return 2;
    if (uniq == 0 || uniq > SCALE) return 3;
    uint16_t *freq = (uint16_t *)calloc(uniq, sizeof(uint16_t));
    uint32_t *cum = (uint32_t *)calloc((size_t)uniq + 1, sizeof(uint32_t));
    uint16_t *idx = (uint16_t *)calloc(n ? n : 1, sizeof(uint16_t));
    uint8_t *renorm = (uint8_t *)malloc((size_t)n * 8u + 16u);
    if (!freq || !cum || !idx || !renorm) return 4;
    if (!read_full(freq, (size_t)uniq * 2u)) return 5;
    if (n && !read_full(idx, (size_t)n * 2u)) return 6;
    uint32_t sum = 0;
    for (uint32_t i = 0; i < uniq; i++) {
        sum += freq[i];
        cum[i + 1] = sum;
    }
    if (sum != SCALE) return 7;
    size_t rcount = 0;
    uint32_t x = RANS_L;
    for (uint32_t k = n; k-- > 0;) {
        uint16_t sym = idx[k];
        if (sym >= uniq || freq[sym] == 0) return 8;
        uint32_t f = freq[sym];
        uint32_t start = cum[sym];
        uint32_t x_max = ((RANS_L >> SCALE_BITS) << 8) * f;
        while (x >= x_max) {
            renorm[rcount++] = (uint8_t)(x & 255u);
            x >>= 8;
        }
        uint32_t q = x / f;
        uint32_t r = x % f;
        uint64_t next = ((uint64_t)q << SCALE_BITS) + (uint64_t)r + (uint64_t)start;
        if (next >= (1ull << 32)) return 9;
        x = (uint32_t)next;
    }
    uint32_t payload_len = 4u + (uint32_t)rcount;
    if (!write_full(&payload_len, 4)) return 10;
    uint8_t xb[4] = {
        (uint8_t)(x & 255u),
        (uint8_t)((x >> 8) & 255u),
        (uint8_t)((x >> 16) & 255u),
        (uint8_t)((x >> 24) & 255u),
    };
    if (!write_full(xb, 4)) return 10;
    for (size_t i = rcount; i-- > 0;) {
        if (!write_full(&renorm[i], 1)) return 10;
    }
    free(freq);
    free(cum);
    free(idx);
    free(renorm);
    return 0;
}

static int decode_main(void) {
    uint32_t n = 0, uniq = 0, payload_len = 0;
    if (!read_full(&n, 4) || !read_full(&uniq, 4) || !read_full(&payload_len, 4)) return 2;
    if (uniq == 0 || uniq > SCALE || payload_len < 4) return 3;
    uint16_t *freq = (uint16_t *)calloc(uniq, sizeof(uint16_t));
    uint32_t *cum = (uint32_t *)calloc((size_t)uniq + 1, sizeof(uint32_t));
    uint8_t *payload = (uint8_t *)malloc(payload_len);
    uint16_t *idx = (uint16_t *)calloc(n ? n : 1, sizeof(uint16_t));
    if (!freq || !cum || !payload || !idx) return 4;
    if (!read_full(freq, (size_t)uniq * 2u)) return 5;
    if (!read_full(payload, payload_len)) return 6;
    uint32_t sum = 0;
    for (uint32_t i = 0; i < uniq; i++) {
        sum += freq[i];
        cum[i + 1] = sum;
    }
    if (sum != SCALE) return 7;
    uint32_t x = (uint32_t)payload[0] | ((uint32_t)payload[1] << 8) |
                 ((uint32_t)payload[2] << 16) | ((uint32_t)payload[3] << 24);
    uint32_t pos = 4;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t cf = x & (SCALE - 1u);
        uint32_t lo = 0, hi = uniq - 1u;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2u;
            if (cum[mid + 1u] <= cf) lo = mid + 1u;
            else hi = mid;
        }
        uint32_t f = freq[lo];
        uint32_t start = cum[lo];
        if (f == 0 || cf < start) return 8;
        uint64_t nx = (uint64_t)f * (x >> SCALE_BITS) + (uint64_t)cf - (uint64_t)start;
        if (nx >= (1ull << 32)) return 9;
        x = (uint32_t)nx;
        while (x < RANS_L) {
            if (pos >= payload_len) return 10;
            x = (x << 8) | payload[pos++];
        }
        idx[i] = (uint16_t)lo;
    }
    if (pos != payload_len) return 11;
    if (!write_full(idx, (size_t)n * 2u)) return 12;
    free(freq);
    free(cum);
    free(payload);
    free(idx);
    return 0;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc != 2) return 1;
    if (strcmp(argv[1], "enc") == 0) return encode_main();
    if (strcmp(argv[1], "dec") == 0) return decode_main();
    return 1;
}
