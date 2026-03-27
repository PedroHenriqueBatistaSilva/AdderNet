/*
 * HDC Core — Hyperdimensional Computing primitives
 * ===================================================
 *   Binary hypervectors of D=10000 bits (157 x uint64_t).
 *   Zero floating-point arithmetic.
 *
 *   Operations:
 *     bind   = XOR bit a bit
 *     bundle = majority vote bit a bit
 *     hamming distance = popcount(XOR)
 *
 *   Compile: included by addernet_hdc.c, or standalone test.
 */

#ifndef HDC_CORE_H
#define HDC_CORE_H

#include <stdint.h>
#include <stdalign.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HDC_DIM
#define HDC_DIM     2500
#endif
#ifndef HDC_WORDS
#define HDC_WORDS   40           /* ceil(2500/64) */
#endif
#define HDC_BYTES   (HDC_WORDS * sizeof(uint64_t))

/* Runtime-configurable dimension support */
#define HDC_WORDS_FOR(dim) (((dim) + 63) / 64)
#define HDC_BYTES_FOR(dim) (HDC_WORDS_FOR(dim) * sizeof(uint64_t))

typedef uint64_t hv_t[HDC_WORDS];

/* Generate a random hypervector (approximately 50% bits set) */
void hv_random(hv_t out);

/* Seed the internal RNG (for reproducibility) */
void hv_seed(unsigned int seed);

/* Get/set the raw RNG state (for save/restore around deterministic encoding) */
uint64_t hv_seed_state(void);
void hv_seed_set(uint64_t state);

/* out = a XOR b */
void hv_bind(hv_t out, const hv_t a, const hv_t b);

/* out = majority vote of n vectors (bundling) */
void hv_bundle(hv_t out, const hv_t *vecs, int n);

/* Hamming distance: number of differing bits */
int  hv_hamming(const hv_t a, const hv_t b);

/* Similarity: 1.0 - hamming/D */
float hv_similarity(const hv_t a, const hv_t b);

/* dst = src */
void hv_copy(hv_t dst, const hv_t src);

/* dst = all zeros */
void hv_zero(hv_t dst);

/* dst = all ones (complement of zero) */
void hv_one(hv_t dst);

/* Random integer in [0, max) */
int  hv_randint(int max);

/* Deterministic hypervector from seed (on-the-fly generation, zero storage) */
void hv_from_seed(hv_t out, uint64_t seed);

/* Melhoria 3: Hadamard/Walsh sequence encoding (orthogonal base vectors) */
void hv_from_hadamard(hv_t out, int var, int bin);

/* Flip bits with given probability (temperature). 0.0 = copy, 1.0 = total noise. */
void hv_add_noise(hv_t out, const hv_t src, float temperature);

/* ---- Cache for hv_from_seed (OPT-1) ---- */
typedef struct {
    uint64_t *enc_cache;
    int table_size;
    int n_vars;
    int cache_ready;
} hdc_cache;

hdc_cache *hdc_cache_create(int n_vars, int table_size);
void hdc_cache_free(hdc_cache *c);
void hdc_cache_build(hdc_cache *c);
void hdc_cache_lookup(const hdc_cache *c, int var, int bin, hv_t out);
int hdc_cache_enabled(const hdc_cache *c);

/* ---- Unrolled Hamming (OPT-6) ---- */
int hv_hamming_unrolled(const hv_t a, const hv_t b);

/* ---- Melhoria 4: AVX2 batch Hamming distance ---- */
/* Computes Hamming distance of 4 queries vs 1 codebook entry simultaneously.
 * dists[0..3] = hamming(queries[0..3], ref).
 * Requires AVX2 + VPOPCNTDQ. */
void hv_hamming_batch4(const hv_t *queries, const hv_t ref, int *dists);

/* ---- Backend detection (OPT-8) ---- */
typedef enum {
    HDC_BACKEND_SCALAR = 0,
    HDC_BACKEND_AVX2,
    HDC_BACKEND_NEON,
} hdc_backend_t;

hdc_backend_t hdc_detect_backend(void);

/* GPU hv_bundle via CUDA (inline PTX, no nvcc required) */
void hv_bundle_cuda(hv_t out, const hv_t *vecs, int n);
void hv_cuda_shutdown(void);

/* ---- Circulant Encoding (PROMPT-1) ---- */
typedef struct {
    hv_t base_val;
    hv_t base_pos;
    int circulant_ready;
} hdc_circulant_t;

void hdc_circulant_init(hdc_circulant_t *circ, uint64_t seed_val, uint64_t seed_pos);
void hv_rotate(hv_t out, const hv_t src, int shift);
void hv_from_circulant(hv_t out, const hdc_circulant_t *circ, int var, int bin);

/* ---- Early Termination (PROMPT-2) ---- */
int hv_hamming_early(const hv_t a, const hv_t b, int best_so_far, int margin);

/* ---- Early-exit Hamming: bail out when dist > max_allowed (Solution 2A) ---- */
int hv_hamming_early_exit(const hv_t a, const hv_t b, int max_allowed);

/* ---- CompHD Dimension Folding (PROMPT-3) ---- */
#ifndef COMPHD_FOLD
#define COMPHD_FOLD 4
#endif
#define HDC_DIM_FOLDED  (HDC_DIM / COMPHD_FOLD)
#define HDC_WORDS_FOLDED ((HDC_DIM_FOLDED + 63) / 64)

typedef uint64_t hv_folded_t[HDC_WORDS_FOLDED];

void hv_fold(hv_folded_t out, const hv_t src, int fold);
int hv_hamming_folded(const hv_folded_t a, const hv_folded_t b);

#ifdef __cplusplus
}
#endif

#endif /* HDC_CORE_H */
