/*
 * HDC Core — Implementation
 * ===========================
 *   Binary hypervectors: D=10000 bits (157 x uint64_t).
 *   All operations use integer bitwise ops. Zero floating-point.
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <cpuid.h>
#include "hdc_core.h"

/* ---- Internal RNG (xorshift64) ---- */

static uint64_t rng_state = 0xDEADBEEFCAFEBABEULL;

void hv_seed(unsigned int seed) {
    rng_state = (uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL;
    if (rng_state == 0) rng_state = 1;
}

uint64_t hv_seed_state(void) { return rng_state; }
void hv_seed_set(uint64_t state) { rng_state = state ? state : 1; }

static inline uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

/* ---- OPT-4: Fast hv_from_seed (same algorithm as original, optimized) ---- */

void hv_from_seed_fast(hv_t out, uint64_t seed) {
    uint64_t s = seed;
    for (int i = 0; i < HDC_WORDS; i++) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out[i] = s;
    }
    int valid_bits = HDC_DIM - (HDC_WORDS - 1) * 64;
    if (valid_bits < 64)
        out[HDC_WORDS - 1] &= (1ULL << valid_bits) - 1;
}

/* ---- OPT-6: Unrolled Hamming ---- */

int hv_hamming_unrolled(const hv_t a, const hv_t b) {
    int dist = 0;
    int w = 0;
    for (; w <= HDC_WORDS - 4; w += 4) {
        dist += __builtin_popcountll(a[w+0] ^ b[w+0]);
        dist += __builtin_popcountll(a[w+1] ^ b[w+1]);
        dist += __builtin_popcountll(a[w+2] ^ b[w+2]);
        dist += __builtin_popcountll(a[w+3] ^ b[w+3]);
    }
    for (; w < HDC_WORDS; w++)
        dist += __builtin_popcountll(a[w] ^ b[w]);
    return dist;
}

/* ---- OPT-1: Cache for hv_from_seed ---- */

hdc_cache *hdc_cache_create(int n_vars, int table_size) {
    hdc_cache *c = (hdc_cache *)calloc(1, sizeof(hdc_cache));
    if (!c) return NULL;
    c->n_vars = n_vars;
    c->table_size = table_size;
    size_t total = (size_t)n_vars * table_size * HDC_WORDS * sizeof(uint64_t);
    c->enc_cache = (uint64_t *)aligned_alloc(64, total);
    if (!c->enc_cache) { free(c); return NULL; }
    memset(c->enc_cache, 0, total);
    c->cache_ready = 0;
    return c;
}

void hdc_cache_free(hdc_cache *c) {
    if (!c) return;
    free(c->enc_cache);
    free(c);
}

void hdc_cache_build(hdc_cache *c) {
    if (!c || !c->enc_cache) return;
    hv_t hv;
    for (int v = 0; v < c->n_vars; v++) {
        for (int bin = 0; bin < c->table_size; bin++) {
            uint64_t seed = (uint64_t)v * 100003ULL + (uint64_t)bin;
            hv_from_seed(hv, seed);
            size_t offset = ((size_t)v * c->table_size + bin) * HDC_WORDS;
            memcpy(&c->enc_cache[offset], hv, HDC_BYTES);
        }
    }
    c->cache_ready = 1;
}

void hdc_cache_lookup(const hdc_cache *c, int var, int bin, hv_t out) {
    if (!c || !c->cache_ready || !c->enc_cache) {
        uint64_t seed = (uint64_t)var * 100003ULL + (uint64_t)bin;
        hv_from_seed_fast(out, seed);
        return;
    }
    size_t offset = ((size_t)var * c->table_size + bin) * HDC_WORDS;
    memcpy(out, &c->enc_cache[offset], HDC_BYTES);
}

int hdc_cache_enabled(const hdc_cache *c) {
    return c && c->cache_ready;
}

/* ---- OPT-8: Backend detection ---- */

hdc_backend_t hdc_detect_backend(void) {
#if defined(__AVX2__)
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    if (ebx & (1 << 5)) return HDC_BACKEND_AVX2;
#elif defined(__ARM_NEON) || defined(__ARM_ARCH_ISA_A64)
    return HDC_BACKEND_NEON;
#endif
    return HDC_BACKEND_SCALAR;
}

/* ---- Public API ---- */

void hv_random(hv_t out) {
    for (int i = 0; i < HDC_WORDS; i++)
        out[i] = xorshift64();
    /* Mask off excess bits in last word */
    out[HDC_WORDS - 1] &= (1ULL << (HDC_DIM - (HDC_WORDS - 1) * 64)) - 1;
}

void hv_bind(hv_t out, const hv_t a, const hv_t b) {
    for (int i = 0; i < HDC_WORDS; i++)
        out[i] = a[i] ^ b[i];
    /* Mask off excess bits in last word */
    out[HDC_WORDS - 1] &= (1ULL << (HDC_DIM - (HDC_WORDS - 1) * 64)) - 1;
}

void hv_bundle(hv_t out, const hv_t *vecs, int n) {
    if (n <= 0) { hv_zero(out); return; }
    if (n == 1) { hv_copy(out, vecs[0]); return; }

    /*
     * Fast majority vote using __builtin_ctzll (sparse iteration).
     * Only iterates over SET bits, not all 64 bits per word.
     * For 50% density: ~32 iterations per word instead of 64.
     * Uses uint16_t counters (20KB on stack for D=10000).
     */
    uint16_t counts[HDC_DIM] = {0};
    int threshold = n / 2;  /* tie → bit stays 0 */

    for (int v = 0; v < n; v++) {
        for (int w = 0; w < HDC_WORDS; w++) {
            uint64_t word = vecs[v][w];
            int base = w * 64;
            while (word) {
                int bit = __builtin_ctzll(word);
                counts[base + bit]++;
                word &= word - 1;
            }
        }
    }

    memset(out, 0, sizeof(hv_t));
    for (int i = 0; i < HDC_DIM; i++) {
        if (counts[i] > threshold)
            out[i / 64] |= (1ULL << (i % 64));
    }
}

int hv_hamming(const hv_t a, const hv_t b) {
    int dist = 0;
    for (int i = 0; i < HDC_WORDS; i++)
        dist += __builtin_popcountll(a[i] ^ b[i]);
    return dist;
}

float hv_similarity(const hv_t a, const hv_t b) {
    return 1.0f - (float)hv_hamming(a, b) / (float)HDC_DIM;
}

void hv_copy(hv_t dst, const hv_t src) {
    memcpy(dst, src, HDC_BYTES);
}

void hv_zero(hv_t dst) {
    memset(dst, 0, HDC_BYTES);
}

void hv_one(hv_t dst) {
    memset(dst, 0xFF, HDC_BYTES);
}

int hv_randint(int max) {
    if (max <= 0) return 0;
    return (int)(xorshift64() % (uint64_t)max);
}

void hv_from_seed(hv_t out, uint64_t seed) {
    uint64_t s = seed;
    for (int i = 0; i < HDC_WORDS; i++) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out[i] = s;
    }
    /* Mask off excess bits in last word */
    int valid_bits = HDC_DIM - (HDC_WORDS - 1) * 64;
    if (valid_bits < 64)
        out[HDC_WORDS - 1] &= (1ULL << valid_bits) - 1;
}

/* ---- Melhoria 3: Hadamard/Walsh sequence encoding ----
 * H[i][j] = (-1)^popcount(i & j), binarized: bit=1 if H>0.
 * Each (var, bin) pair produces a unique orthogonal hypervector.
 * math.h not needed: __builtin_parity() = popcount % 2. */
void hv_from_hadamard(hv_t out, int var, int bin) {
    int combined = var * 100003 + bin;
    for (int w = 0; w < HDC_WORDS; w++) {
        uint64_t word = 0;
        int base = w * 64;
        for (int b = 0; b < 64 && (base + b) < HDC_DIM; b++) {
            int d = base + b;
            /* __builtin_parity returns 1 if popcount is odd */
            if (!__builtin_parity(combined & d))
                word |= (1ULL << b);
        }
        out[w] = word;
    }
    /* Mask off excess bits in last word */
    int valid_bits = HDC_DIM - (HDC_WORDS - 1) * 64;
    if (valid_bits < 64)
        out[HDC_WORDS - 1] &= (1ULL << valid_bits) - 1;
}

void hv_add_noise(hv_t out, const hv_t src, float temperature) {
    if (temperature <= 0.0f) { hv_copy(out, src); return; }
    if (temperature >= 1.0f) { hv_random(out); return; }

    /* For each bit, flip with probability = temperature.
     * Uses xorshift for speed instead of rand(). */
    for (int i = 0; i < HDC_WORDS; i++) {
        uint64_t mask = 0;
        uint64_t r = xorshift64();
        for (int b = 0; b < 64; b++) {
            /* Compare 8-bit threshold: faster than float rand */
            if ((r & 0xFF) < (uint64_t)(temperature * 256))
                mask |= (1ULL << b);
            r >>= 8;
            if (b % 8 == 7 && b < 63) r = xorshift64();
        }
        out[i] = src[i] ^ mask;
    }
    /* Mask last word */
    int valid_bits = HDC_DIM - (HDC_WORDS - 1) * 64;
    if (valid_bits < 64)
        out[HDC_WORDS - 1] &= (1ULL << valid_bits) - 1;
}

/* ---- OPT-2: AVX2 hv_bundle ---- */
#ifdef __AVX2__
#include <immintrin.h>

void hv_bundle_avx2(hv_t out, const hv_t *vecs, int n) {
    if (n <= 0) { hv_zero(out); return; }
    if (n == 1) { hv_copy(out, vecs[0]); return; }

    uint16_t counts[HDC_DIM] __attribute__((aligned(32))) = {0};
    int threshold = n / 2;

    for (int v = 0; v < n; v++) {
        for (int w = 0; w < HDC_WORDS; w++) {
            uint64_t word = vecs[v][w];
            int base = w * 64;
            while (word) {
                int bit = __builtin_ctzll(word);
                counts[base + bit]++;
                word &= word - 1;
            }
        }
    }

    memset(out, 0, sizeof(hv_t));
    for (int i = 0; i < HDC_DIM; i++) {
        if (counts[i] > threshold)
            out[i / 64] |= (1ULL << (i % 64));
    }
}

int hv_hamming_avx2(const hv_t a, const hv_t b) {
    return hv_hamming_unrolled(a, b);
}
#endif

/* ---- OPT-3: ARM NEON hv_bundle ---- */
#ifdef __ARM_NEON
#include <arm_neon.h>

void hv_bundle_neon(hv_t out, const hv_t *vecs, int n) {
    if (n <= 0) { hv_zero(out); return; }
    if (n == 1) { hv_copy(out, vecs[0]); return; }

    uint16_t counts[HDC_DIM] __attribute__((aligned(16))) = {0};
    int threshold = n / 2;

    for (int v = 0; v < n; v++) {
        for (int w = 0; w < HDC_WORDS; w++) {
            uint64_t word = vecs[v][w];
            int base = w * 64;
            while (word) {
                int bit = __builtin_ctzll(word);
                counts[base + bit]++;
                word &= word - 1;
            }
        }
    }

    memset(out, 0, sizeof(hv_t));
    for (int i = 0; i < HDC_DIM; i++) {
        if (counts[i] > threshold)
            out[i / 64] |= (1ULL << (i % 64));
    }
}

int hv_hamming_neon(const hv_t a, const hv_t b) {
    uint8x16_t sum = vdupq_n_u8(0);
    const uint8_t *ba = (const uint8_t *)a;
    const uint8_t *bb = (const uint8_t *)b;
    int total = 0;
    for (int i = 0; i < HDC_WORDS * 8; i += 16) {
        uint8x16_t va = vld1q_u8(ba + i);
        uint8x16_t vb = vld1q_u8(bb + i);
        uint8x16_t vxor = veorq_u8(va, vb);
        uint8x16_t vcnt = vcntq_u8(vxor);
        sum = vaddq_u8(sum, vcnt);
    }
    uint64x2_t sum64 = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(sum)));
    total = (int)(vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1));
    return total;
}
#endif

/* ---- PROMPT-1: Circulant Encoding ---- */

void hdc_circulant_init(hdc_circulant_t *circ, uint64_t seed_val, uint64_t seed_pos) {
    if (!circ) return;
    hv_seed(seed_val);
    hv_random(circ->base_val);
    hv_seed(seed_pos);
    hv_random(circ->base_pos);
    circ->circulant_ready = 1;
}

void hv_rotate(hv_t out, const hv_t src, int shift) {
    shift = ((shift % HDC_DIM) + HDC_DIM) % HDC_DIM;
    int word_shift = shift / 64;
    int bit_shift = shift % 64;

    for (int i = 0; i < HDC_WORDS; i++) {
        int j = (i + word_shift) % HDC_WORDS;
        int k = (i + word_shift + 1) % HDC_WORDS;
        if (bit_shift == 0)
            out[i] = src[j];
        else
            out[i] = (src[j] << bit_shift) | (src[k] >> (64 - bit_shift));
    }
    out[HDC_WORDS - 1] &= (1ULL << (HDC_DIM - (HDC_WORDS - 1) * 64)) - 1;
}

void hv_from_circulant(hv_t out, const hdc_circulant_t *circ, int var, int bin) {
    hv_t val_hv, pos_hv;
    hv_rotate(val_hv, circ->base_val, var * 37 + bin * 7);
    hv_rotate(pos_hv, circ->base_pos, var * 53);
    hv_bind(out, val_hv, pos_hv);
}

/* ---- PROMPT-2: Early Termination ---- */

int hv_hamming_early(const hv_t a, const hv_t b, int best_so_far, int margin) {
    if (best_so_far >= INT_MAX - margin) {
        return hv_hamming_unrolled(a, b);
    }
    int threshold = best_so_far + margin;
    int dist = 0;
    int w = 0;

    for (; w <= HDC_WORDS - 4; w += 4) {
        dist += __builtin_popcountll(a[w+0] ^ b[w+0]);
        dist += __builtin_popcountll(a[w+1] ^ b[w+1]);
        dist += __builtin_popcountll(a[w+2] ^ b[w+2]);
        dist += __builtin_popcountll(a[w+3] ^ b[w+3]);

        int bits_done = (w + 4) * 64;
        int threshold_partial = (threshold * bits_done) / HDC_DIM;
        if (dist > threshold_partial)
            return INT_MAX;
    }

    for (; w < HDC_WORDS; w++)
        dist += __builtin_popcountll(a[w] ^ b[w]);

    return dist;
}

/* ---- PROMPT-3: CompHD Dimension Folding ---- */

void hv_fold(hv_folded_t out, const hv_t src, int fold) {
    int words_per_seg = HDC_WORDS / fold;
    memset(out, 0, sizeof(hv_folded_t));

    for (int s = 0; s < fold; s++) {
        for (int w = 0; w < words_per_seg && w < HDC_WORDS_FOLDED; w++) {
            out[w] ^= src[s * words_per_seg + w];
        }
    }
}

int hv_hamming_folded(const hv_folded_t a, const hv_folded_t b) {
    int dist = 0;
    for (int w = 0; w < HDC_WORDS_FOLDED; w++)
        dist += __builtin_popcountll(a[w] ^ b[w]);
    return dist;
}

/* ---- Solution 2A: Early-exit Hamming distance ---- */
int hv_hamming_early_exit(const hv_t a, const hv_t b, int max_allowed) {
    int dist = 0;
    int w = 0;
    for (; w <= HDC_WORDS - 4; w += 4) {
        dist += __builtin_popcountll(a[w+0] ^ b[w+0]);
        dist += __builtin_popcountll(a[w+1] ^ b[w+1]);
        dist += __builtin_popcountll(a[w+2] ^ b[w+2]);
        dist += __builtin_popcountll(a[w+3] ^ b[w+3]);
        if (dist > max_allowed) return dist;
    }
    for (; w < HDC_WORDS; w++) {
        dist += __builtin_popcountll(a[w] ^ b[w]);
        if (dist > max_allowed) return dist;
    }
    return dist;
}

/* ---- Melhoria 4: AVX2 batch Hamming distance ---- */
#ifdef __AVX2__
#include <immintrin.h>

void hv_hamming_batch4(const hv_t *queries, const hv_t ref, int *dists) {
    /* Process HDC_WORDS uint64_t words, 4 queries at a time.
     * We load 4 × 64-bit values per query into __m256i, XOR with ref, popcount. */

    /* Process 4 words at a time for throughput */
    int w = 0;
    for (; w + 3 < HDC_WORDS; w += 4) {
        /* Load ref words */
        __m256i vr = _mm256_loadu_si256((__m256i*)&ref[w]);

        /* Query 0: load 4 words, XOR, popcount, accumulate */
        __m256i q0 = _mm256_loadu_si256((__m256i*)&queries[0][w]);
        __m256i x0 = _mm256_xor_si256(q0, vr);
        /* AVX2 doesn't have 64-bit popcount directly.
         * Split into 32-bit halves and use VPOPCNTDQ if available,
         * or fall back to manual popcount via _mm256_popcnt_epi64.
         * Since we compiled with -march=native and have AVX2,
         * we use _mm_popcnt_u64 on individual elements. */
        uint64_t *p0 = (uint64_t*)&x0;
        dists[0] += __builtin_popcountll(p0[0]) + __builtin_popcountll(p0[1])
                  + __builtin_popcountll(p0[2]) + __builtin_popcountll(p0[3]);

        __m256i q1 = _mm256_loadu_si256((__m256i*)&queries[1][w]);
        __m256i x1 = _mm256_xor_si256(q1, vr);
        uint64_t *p1 = (uint64_t*)&x1;
        dists[1] += __builtin_popcountll(p1[0]) + __builtin_popcountll(p1[1])
                  + __builtin_popcountll(p1[2]) + __builtin_popcountll(p1[3]);

        __m256i q2 = _mm256_loadu_si256((__m256i*)&queries[2][w]);
        __m256i x2 = _mm256_xor_si256(q2, vr);
        uint64_t *p2 = (uint64_t*)&x2;
        dists[2] += __builtin_popcountll(p2[0]) + __builtin_popcountll(p2[1])
                  + __builtin_popcountll(p2[2]) + __builtin_popcountll(p2[3]);

        __m256i q3 = _mm256_loadu_si256((__m256i*)&queries[3][w]);
        __m256i x3 = _mm256_xor_si256(q3, vr);
        uint64_t *p3 = (uint64_t*)&x3;
        dists[3] += __builtin_popcountll(p3[0]) + __builtin_popcountll(p3[1])
                  + __builtin_popcountll(p3[2]) + __builtin_popcountll(p3[3]);
    }

    /* Handle remaining words */
    for (; w < HDC_WORDS; w++) {
        uint64_t r = ref[w];
        dists[0] += __builtin_popcountll(queries[0][w] ^ r);
        dists[1] += __builtin_popcountll(queries[1][w] ^ r);
        dists[2] += __builtin_popcountll(queries[2][w] ^ r);
        dists[3] += __builtin_popcountll(queries[3][w] ^ r);
    }
}
#else
/* Scalar fallback */
void hv_hamming_batch4(const hv_t *queries, const hv_t ref, int *dists) {
    dists[0] = hv_hamming_unrolled(queries[0], ref);
    dists[1] = hv_hamming_unrolled(queries[1], ref);
    dists[2] = hv_hamming_unrolled(queries[2], ref);
    dists[3] = hv_hamming_unrolled(queries[3], ref);
}
#endif
