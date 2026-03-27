#include "hdc_lsh.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

hdc_lsh_t* hdc_lsh_build_ex(const hv_t *codebook, int n_classes, int k, int l) {
    if (k > LSH_MAX_K || l > LSH_MAX_L || k <= 0 || l <= 0) return NULL;
    
    hdc_lsh_t *lsh = (hdc_lsh_t *)calloc(1, sizeof(hdc_lsh_t));
    if (!lsh) return NULL;
    lsh->n_classes = n_classes;
    lsh->k = k;
    lsh->l = l;
    
    int n_buckets = 1 << k;

    for (int tbl = 0; tbl < l; tbl++) {
        for (int bit = 0; bit < k; bit++) {
            lsh->bit_indices[tbl][bit] = rand() % HDC_DIM;
        }
        lsh->buckets[tbl] = (int **)calloc(n_buckets, sizeof(int *));
        lsh->bucket_sizes[tbl] = (int *)calloc(n_buckets, sizeof(int));
    }

    for (int c = 0; c < n_classes; c++) {
        for (int tbl = 0; tbl < l; tbl++) {
            unsigned int hash = 0;
            for (int bit = 0; bit < k; bit++) {
                int bit_idx = lsh->bit_indices[tbl][bit];
                int word_idx = bit_idx >> 6;
                int bit_pos = bit_idx & 63;
                int bit_val = (codebook[c][word_idx] >> bit_pos) & 1U;
                hash |= (bit_val << bit);
            }
            int sz = lsh->bucket_sizes[tbl][hash];
            int *new_bucket = (int *)realloc(lsh->buckets[tbl][hash], (sz + 1) * sizeof(int));
            if (new_bucket) {
                lsh->buckets[tbl][hash] = new_bucket;
                lsh->buckets[tbl][hash][sz] = c;
                lsh->bucket_sizes[tbl][hash]++;
            }
        }
    }
    return lsh;
}

int hdc_lsh_query(const hdc_lsh_t *lsh, const hv_t query,
                  int *out_candidates, int max_candidates) {
    char *seen = (char *)calloc(lsh->n_classes, sizeof(char));
    if (!seen) return 0;
    
    int n_candidates = 0;
    int k = lsh->k;
    int l = lsh->l;
    int n_buckets = 1 << k;

    for (int tbl = 0; tbl < l; tbl++) {
        unsigned int hash = 0;
        for (int bit = 0; bit < k; bit++) {
            int bit_idx = lsh->bit_indices[tbl][bit];
            int word_idx = bit_idx >> 6;
            int bit_pos = bit_idx & 63;
            int bit_val = (query[word_idx] >> bit_pos) & 1U;
            hash |= (bit_val << bit);
        }

        for (int i = 0; i < lsh->bucket_sizes[tbl][hash]; i++) {
            int c = lsh->buckets[tbl][hash][i];
            if (!seen[c] && n_candidates < max_candidates) {
                seen[c] = 1;
                out_candidates[n_candidates++] = c;
            }
        }
    }
    free(seen);
    return n_candidates;
}

void hdc_lsh_free(hdc_lsh_t *lsh) {
    if (!lsh) return;
    int n_buckets = 1 << lsh->k;
    for (int tbl = 0; tbl < lsh->l; tbl++) {
        if (lsh->buckets[tbl]) {
            for (int b = 0; b < n_buckets; b++) {
                free(lsh->buckets[tbl][b]);
            }
            free(lsh->buckets[tbl]);
        }
        free(lsh->bucket_sizes[tbl]);
    }
    free(lsh);
}
