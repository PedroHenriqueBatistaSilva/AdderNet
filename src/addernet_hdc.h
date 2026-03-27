/*
 * AdderNet-HDC — Multivariate classification using Hyperdimensional Computing
 * ============================================================================
 *   Combines AdderNet encoding (discretization via lookup) with HDC
 *   (binding=XOR, bundling=majority vote, search=Hamming distance).
 *
 *   Zero floating-point multiplication at inference.
 */

#ifndef ADDERNET_HDC_H
#define ADDERNET_HDC_H

#include "hdc_core.h"
#include "hdc_lsh.h"

/* Interaction pair for feature interaction encoding (Problem 6) */
typedef struct {
    int i;
    int j;
} an_interaction_pair_t;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Multivariate HDC model.
 *
 *   position_hvs[v]   — fixed random hypervector for variable v (role vector)
 *   codebook[c]        — bundled prototype for class c
 *
 * Value vectors are generated on-the-fly via hv_from_seed (zero storage).
 *
 * Inference flow (position-value encoding):
 *   H_val = hv_from_seed(var=v, bin=idx)                    "value X"
 *   pair_v = bind(position_hvs[v], H_val)                   "var v has value X"
 *   H_query = bundle(pair_0, pair_1, ..., pair_{n-1})
 *   class = argmin_c  hamming(H_query, codebook[c])
 */
typedef struct {
    int        n_vars;
    int        n_classes;
    int        table_size;         /* encoding resolution (power of 2) */
    int        table_mask;         /* table_size - 1 */
    int       *bias;               /* bias[v] per variable */
    hv_t     *position_hvs;       /* [n_vars] role vectors (one per variable) */
    hv_t     *codebook;           /* [n_classes] class prototypes */
    char     **class_names;        /* [n_classes] or NULL */
    hdc_cache *cache;              /* OPT-1: encoding cache, NULL if disabled */
    int        use_cache;          /* 1 = use cache, 0 = compute on-the-fly */
    int        n_threads;          /* OPT-5: threads for batch prediction */
    
    /* PROMPT-1: Circulant encoding */
    hdc_circulant_t circulant;
    int               use_circulant;
    
    /* Melhoria 3: Hadamard encoding */
    int               use_hadamard;
    
    /* PROMPT-2: Early termination */
    int  use_early_term;
    int  early_term_margin;
    
    /* PROMPT-3: CompHD dimension folding */
    hv_folded_t *codebook_folded;
    int          use_folded;
    int          fold_factor;
    
    /* LSH: Locality-Sensitive Hashing for large codebook search */
    hdc_lsh_t *lsh_index;
    int         use_lsh;

    /* Problem 6: Feature interaction encoding */
    an_interaction_pair_t *interaction_pairs;
    int n_interaction_pairs;
} an_hdc_model;

/*
 * an_hdc_create — Allocate a model.
 *
 *   n_vars:     number of input variables
 *   n_classes:  number of output classes
 *   table_size: encoding table size per variable (must be power of 2, e.g. 256)
 *   bias:       array of n_vars bias values (or NULL for auto: table_size/2)
 *
 *   Allocates random hypervectors for the encoding table.
 *   Returns: pointer to new model, or NULL on failure.
 */
an_hdc_model *an_hdc_create(int n_vars, int n_classes, int table_size,
                            const int *bias);

/*
 * an_hdc_free — Free a model.
 */
void an_hdc_free(an_hdc_model *m);

/*
 * an_hdc_train — Train the codebook from labeled data.
 *
 *   X:          n_samples × n_vars matrix (row-major doubles)
 *   y:          n_samples class labels (int, 0-indexed)
 *   n_samples:  number of training samples
 *
 *   For each class, encodes all its samples into hypervectors,
 *   then bundles them into the class prototype.
 */
void an_hdc_train(an_hdc_model *m, const double *X, const int *y, int n_samples);

/*
 * an_hdc_retrain — Iterative retraining with RefineHD margin (Melhoria 1)
 *                  and NeuralHD dimension regeneration (Melhoria 2).
 *                  Internal 80/20 split for validation-based early stopping.
 *
 *   X:          n_samples × n_vars matrix (row-major doubles)
 *   y:          n_samples class labels (int, 0-indexed)
 *   n_samples:  number of training samples
 *   n_iter:     max number of correction iterations
 *   lr:         learning rate (1.0 = standard)
 *   margin:     RefineHD margin as absolute Hamming distance (int)
 *   regen_rate: NeuralHD dimension regeneration rate (0.0 = off, 0.02-0.05 recommended)
 *   patience:   early stopping patience (0 = disabled)
 *   verbose:    0=silent, N=print every N epochs
 *   epochs_run: output — epochs actually executed (NULL to ignore)
 *
 *   Returns: number of epochs actually run.
 *   Early stopping monitors VALIDATION accuracy (last 20% of data), never training accuracy.
 */
int an_hdc_retrain(an_hdc_model *m, const double *X, const int *y,
                    int n_samples, int n_iter, float lr,
                    int margin, float regen_rate, int patience,
                    int verbose, int *epochs_run);

/*
 * an_hdc_predict — Classify one sample. Zero multiplication.
 *
 *   x: array of n_vars input values
 *   Returns: predicted class label (0-indexed), or -1 on error.
 */
int an_hdc_predict(const an_hdc_model *m, const double *x);

/*
 * an_hdc_predict_batch — Classify n samples.
 *
 *   X:       n × n_vars matrix (row-major doubles)
 *   outputs: array of n ints (pre-allocated by caller)
 *   Returns: 0 on success, -1 on error.
 */
int an_hdc_predict_batch(const an_hdc_model *m, const double *X,
                         int *outputs, int n);

/*
 * an_hdc_predict_batch_avx — Classify n samples using AVX2 SIMD (Melhoria 4).
 *   Processes 4 samples simultaneously for Hamming distance computation.
 *   Falls back to scalar if AVX2 not available.
 *   Returns: 0 on success, -1 on error.
 */
int an_hdc_predict_batch_avx(const an_hdc_model *m, const double *X,
                              int *outputs, int n);

/*
 * an_hdc_predict_batch_mt — Classify n samples using multiple threads (OPT-5).
 *
 *   X:       n × n_vars matrix (row-major doubles)
 *   outputs: array of n ints (pre-allocated by caller)
 *   n:       number of samples
 *   n_threads: number of threads (0 = auto-detect)
 *   Returns: 0 on success, -1 on error.
 */
int an_hdc_predict_batch_mt(const an_hdc_model *m, const double *X,
                            int *outputs, int n, int n_threads);

/*
 * an_hdc_warm_cache — Pre-compute encoding cache (OPT-1).
 *   Call once before benchmarking.
 */
void an_hdc_warm_cache(an_hdc_model *m);

/*
 * an_hdc_set_cache — Enable/disable encoding cache (OPT-1).
 */
void an_hdc_set_cache(an_hdc_model *m, int use_cache);

/*
 * an_hdc_set_threads — Set thread count for batch prediction (OPT-5).
 */
void an_hdc_set_threads(an_hdc_model *m, int n_threads);

/*
 * an_hdc_save — Serialize model to binary file.
 * Returns: 0 on success, -1 on error.
 */
int an_hdc_save(const an_hdc_model *m, const char *path);

/*
 * an_hdc_load — Deserialize model from binary file.
 * Returns: pointer to new model, or NULL on failure.
 */
an_hdc_model *an_hdc_load(const char *path);

/*
 * an_hdc_predict_batch_cuda — Batch prediction on GPU.
 * N predictions in parallel. Falls back to CPU if CUDA unavailable.
 * Returns: 0 on success, -1 on error.
 */
int an_hdc_predict_batch_cuda(an_hdc_model *m, const double *X,
                               int *y_pred, int N);

/* PROMPT-1: Circulant encoding control */
void an_hdc_set_circulant(an_hdc_model *m, int enable);

/* Melhoria 3: Hadamard encoding control */
void an_hdc_set_hadamard(an_hdc_model *m, int enable);

/* PROMPT-2: Early termination control */
void an_hdc_set_early_termination(an_hdc_model *m, int enable, int margin);

/* PROMPT-3: CompHD dimension folding */
void an_hdc_fold_codebook(an_hdc_model *m, int fold);
int an_hdc_predict_folded(const an_hdc_model *m, const double *x);

/* LSH: Locality-Sensitive Hashing */
void an_hdc_build_lsh(an_hdc_model *m);
void an_hdc_build_lsh_ex(an_hdc_model *m, int k, int l);
void an_hdc_set_lsh(an_hdc_model *m, int enable);

/* predict_top_k: Retorna os K classes mais próximas */
void an_hdc_predict_top_k(const an_hdc_model *m, const double *x,
                          int *out_classes, int k);

/* Problem 6: Set interaction pairs for feature interaction encoding */
void an_hdc_set_interactions(an_hdc_model *m, const int *pairs_i,
                              const int *pairs_j, int n_pairs);

#ifdef __cplusplus
}
#endif

#endif /* ADDERNET_HDC_H */
