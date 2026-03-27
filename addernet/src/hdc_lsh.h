#ifndef HDC_LSH_H
#define HDC_LSH_H

#include "hdc_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LSH_K
#define LSH_K        16
#endif
#ifndef LSH_L
#define LSH_L         4
#endif
#define LSH_MAX_K     16
#define LSH_MAX_L     16
#define LSH_BUCKETS   (1 << LSH_K)

typedef struct {
    int k;
    int l;
    int bit_indices[LSH_MAX_L][LSH_MAX_K];
    int **buckets[LSH_MAX_L];
    int  *bucket_sizes[LSH_MAX_L];
    int   n_classes;
} hdc_lsh_t;

hdc_lsh_t* hdc_lsh_build_ex(const hv_t *codebook, int n_classes, int k, int l);
#define hdc_lsh_build(codebook, n_classes) hdc_lsh_build_ex(codebook, n_classes, LSH_K, LSH_L)

int hdc_lsh_query(const hdc_lsh_t *lsh, const hv_t query,
                  int *out_candidates, int max_candidates);
void hdc_lsh_free(hdc_lsh_t *lsh);

#ifdef __cplusplus
}
#endif

#endif
