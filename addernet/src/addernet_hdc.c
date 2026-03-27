/*
 * AdderNet-HDC — Implementation
 * ===============================
 *   Multivariate classification using Hyperdimensional Computing.
 *   Zero floating-point multiplication at inference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#include <unistd.h>
#include "addernet_hdc.h"

/* ---- Internal helpers ---- */

/* Encode a single input value to its hypervector for a given variable.
 * Uses deterministic hv_from_seed — zero storage, computed on the fly. */
static inline void an_hdc_encode(const an_hdc_model *m, int var, double val, hv_t out) {
    int bin = ((int)val + m->bias[var]) & m->table_mask;
    if (m->use_hadamard) {
        hv_from_hadamard(out, var, bin);
    } else if (m->use_circulant && m->circulant.circulant_ready) {
        hv_from_circulant(out, &m->circulant, var, bin);
    } else if (m->use_cache && m->cache && m->cache->cache_ready) {
        hdc_cache_lookup(m->cache, var, bin, out);
    } else {
        uint64_t seed = (uint64_t)var * 100003ULL + (uint64_t)bin;
        hv_from_seed(out, seed);
    }
}

/* ---- Public API ---- */

an_hdc_model *an_hdc_create(int n_vars, int n_classes, int table_size,
                            const int *bias)
{
    if (n_vars <= 0 || n_classes <= 0 || table_size <= 0)
        return NULL;
    if ((table_size & (table_size - 1)) != 0)
        return NULL;  /* must be power of 2 */

    an_hdc_model *m = (an_hdc_model *)calloc(1, sizeof(an_hdc_model));
    if (!m) return NULL;

    m->n_vars     = n_vars;
    m->n_classes  = n_classes;
    m->table_size = table_size;
    m->table_mask = table_size - 1;
    m->use_cache  = 0;
    m->cache      = NULL;
    m->n_threads  = 0;
    m->use_circulant = 0;
    m->circulant.circulant_ready = 0;
    m->use_hadamard = 0;
    m->use_early_term = 0;
    m->early_term_margin = 50;
    m->codebook_folded = NULL;
    m->use_folded = 0;
    m->fold_factor = COMPHD_FOLD;
    m->lsh_index = NULL;
    m->use_lsh = 0;
    m->interaction_pairs = NULL;
    m->n_interaction_pairs = 0;

    /* Bias per variable */
    m->bias = (int *)malloc(n_vars * sizeof(int));
    if (!m->bias) goto fail;
    if (bias) {
        memcpy(m->bias, bias, n_vars * sizeof(int));
    } else {
        for (int v = 0; v < n_vars; v++)
            m->bias[v] = table_size / 2;
    }

    /* Position hypervectors: one random HV per variable (role vector). */
    m->position_hvs = (hv_t *)aligned_alloc(64, n_vars * sizeof(hv_t));
    if (!m->position_hvs) goto fail;
    for (int v = 0; v < n_vars; v++)
        hv_random(m->position_hvs[v]);

    /* Codebook: n_classes prototypes (zeroed, trained later) */
    m->codebook = (hv_t *)aligned_alloc(64, n_classes * sizeof(hv_t));
    if (!m->codebook) goto fail;
    for (int c = 0; c < n_classes; c++)
        hv_zero(m->codebook[c]);

    /* Class names */
    m->class_names = (char **)calloc(n_classes, sizeof(char *));

    return m;

fail:
    an_hdc_free(m);
    return NULL;
}

void an_hdc_free(an_hdc_model *m) {
    if (!m) return;
    free(m->bias);
    free(m->position_hvs);
    free(m->codebook);
    if (m->cache) hdc_cache_free(m->cache);
    if (m->codebook_folded) free(m->codebook_folded);
    if (m->lsh_index) hdc_lsh_free(m->lsh_index);
    if (m->interaction_pairs) free(m->interaction_pairs);
    if (m->class_names) {
        for (int c = 0; c < m->n_classes; c++)
            free(m->class_names[c]);
        free(m->class_names);
    }
    free(m);
}

void an_hdc_train(an_hdc_model *m, const double *X, const int *y, int n_samples) {
    if (!m || !X || !y || n_samples <= 0) return;

    /* Allocate temp storage per class */
    hv_t **class_hvs = (hv_t **)malloc(m->n_classes * sizeof(hv_t *));
    int   *class_cnt = (int *)calloc(m->n_classes, sizeof(int));
    if (!class_hvs || !class_cnt) { free(class_hvs); free(class_cnt); return; }

    /* Count samples per class first */
    int *counts = (int *)calloc(m->n_classes, sizeof(int));
    for (int i = 0; i < n_samples; i++) {
        int c = y[i];
        if (c >= 0 && c < m->n_classes) counts[c]++;
    }

    /* Allocate per-class arrays */
    for (int c = 0; c < m->n_classes; c++) {
        class_hvs[c] = (counts[c] > 0)
            ? (hv_t *)aligned_alloc(64, counts[c] * sizeof(hv_t))
            : NULL;
    }
    free(counts);

    /* Encode all samples: bind(position, value) for each var, then bundle all pairs */
    hv_t *pairs = (hv_t *)aligned_alloc(64,
        (m->n_vars + m->n_interaction_pairs > 0 ? m->n_vars + m->n_interaction_pairs : 1) * sizeof(hv_t));
    hv_t sample_hv;
    for (int i = 0; i < n_samples; i++) {
        int c = y[i];
        if (c < 0 || c >= m->n_classes) continue;

        /* For each variable: pair[i] = bind(position[i], value[i]) */
        for (int v = 0; v < m->n_vars; v++) {
            if (m->use_hadamard) {
                /* Hadamard: direct encoding without position binding */
                int bin = ((int)X[i * m->n_vars + v] + m->bias[v]) & m->table_mask;
                hv_from_hadamard(pairs[v], v, bin);
            } else if (m->use_circulant && m->circulant.circulant_ready) {
                int bin = ((int)X[i * m->n_vars + v] + m->bias[v]) & m->table_mask;
                hv_from_circulant(pairs[v], &m->circulant, v, bin);
            } else {
                hv_t val_hv;
                an_hdc_encode(m, v, X[i * m->n_vars + v], val_hv);
                hv_bind(pairs[v], m->position_hvs[v], val_hv);
            }
        }

        /* Problem 6: Interaction pair encoding (XOR binding of correlated features) */
        for (int k = 0; k < m->n_interaction_pairs; k++) {
            int pi = m->interaction_pairs[k].i;
            int pj = m->interaction_pairs[k].j;
            hv_t hv_i, hv_j;
            an_hdc_encode(m, pi, X[i * m->n_vars + pi], hv_i);
            an_hdc_encode(m, pj, X[i * m->n_vars + pj], hv_j);
            hv_bind(pairs[m->n_vars + k], hv_i, hv_j);
        }

        /* Bundle all pairs into one sample hypervector */
        hv_bundle(sample_hv, pairs, m->n_vars + m->n_interaction_pairs);

        /* Store in class array */
        if (class_hvs[c]) {
            hv_copy(class_hvs[c][class_cnt[c]], sample_hv);
            class_cnt[c]++;
        }
    }
    free(pairs);

    /* OnlineHD: weighted bundling per class to prevent model saturation.
     * Each sample weighted by (1 - similarity to class mean).
     * Redundant samples get low weight, novel samples get high weight. */
    for (int c = 0; c < m->n_classes; c++) {
        if (class_cnt[c] <= 0 || !class_hvs[c]) continue;

        int n_c = class_cnt[c];

        /* First pass: compute unweighted mean for similarity reference */
        hv_t mean_hv;
        hv_bundle(mean_hv, class_hvs[c], n_c);

        /* Compute per-sample weights based on novelty */
        float *weights = (float *)malloc(n_c * sizeof(float));
        if (!weights) {
            /* Fallback: unweighted bundle */
            hv_bundle(m->codebook[c], class_hvs[c], n_c);
            free(class_hvs[c]);
            continue;
        }

        float w_sum = 0;
        for (int s = 0; s < n_c; s++) {
            int ham = hv_hamming_unrolled(class_hvs[c][s], mean_hv);
            float sim = 1.0f - (float)ham / (float)HDC_DIM;
            float w = 1.0f - sim;  /* novelty weight: 0..1 */
            if (w < 0.05f) w = 0.05f;  /* minimum weight to retain contribution */
            weights[s] = w;
            w_sum += w;
        }

        /* Weighted majority vote using bit-count accumulation */
        uint16_t *bit_counts = (uint16_t *)calloc(HDC_DIM, sizeof(uint16_t));
        if (!bit_counts) {
            hv_bundle(m->codebook[c], class_hvs[c], n_c);
            free(weights);
            free(class_hvs[c]);
            continue;
        }

        for (int s = 0; s < n_c; s++) {
            uint16_t w_int = (uint16_t)(weights[s] * 256.0f);
            for (int w = 0; w < HDC_WORDS; w++) {
                uint64_t word = class_hvs[c][s][w];
                int base = w * 64;
                while (word) {
                    int bit = __builtin_ctzll(word);
                    bit_counts[base + bit] += w_int;
                    word &= word - 1;
                }
            }
        }

        /* Threshold: half of total weighted count */
        uint16_t threshold = (uint16_t)(w_sum * 128.0f);  /* w_sum * 256 / 2 */

        hv_zero(m->codebook[c]);
        for (int i = 0; i < HDC_DIM; i++) {
            if (bit_counts[i] > threshold)
                m->codebook[c][i / 64] |= (1ULL << (i % 64));
        }

        free(bit_counts);
        free(weights);
        free(class_hvs[c]);
    }

    free(class_hvs);
    free(class_cnt);
}

/* ---- Problema 1+4: AdaptHD + RefineHD margin + NeuralHD regeneration ----
 *    Internal 80/20 validation split. Early stopping monitors VALIDATION accuracy.
 *    Margin is absolute Hamming distance (int), converted by Python from float. */
int an_hdc_retrain(an_hdc_model *m, const double *X, const int *y,
                    int n_samples, int n_iter, float lr,
                    int margin, float regen_rate, int patience,
                    int verbose, int *epochs_run_out)
{
    if (!m || !X || !y || n_samples <= 0 || n_iter <= 0) {
        if (epochs_run_out) *epochs_run_out = 0;
        return 0;
    }
    if (lr <= 0.0f) lr = 1.0f;
    if (margin < 0) margin = 0;
    if (regen_rate < 0.0f) regen_rate = 0.0f;
    if (regen_rate > 0.03f) regen_rate = 0.03f;  /* Cap at 3% to allow convergence */
    if (patience < 0) patience = 0;

    /* ---- Problema 1: Internal validation split ---- */
    int n_val = n_samples / 4;       /* 25% validation for stability */
    int n_train = n_samples - n_val; /* 75% training */
    if (n_val < 1 && n_samples >= 2) { n_val = 1; n_train = n_samples - 1; }

    /* Encode ALL samples once (train + val) */
    hv_t *sample_hvs = (hv_t *)aligned_alloc(64, n_samples * sizeof(hv_t));
    if (!sample_hvs) { if (epochs_run_out) *epochs_run_out = 0; return 0; }

    int total_pairs = m->n_vars + m->n_interaction_pairs;
    if (total_pairs < 1) total_pairs = 1;
    hv_t *pairs = (hv_t *)aligned_alloc(64, total_pairs * sizeof(hv_t));
    if (!pairs) { free(sample_hvs); if (epochs_run_out) *epochs_run_out = 0; return 0; }

    for (int i = 0; i < n_samples; i++) {
        int c = y[i];
        if (c < 0 || c >= m->n_classes) { hv_zero(sample_hvs[i]); continue; }
        for (int v = 0; v < m->n_vars; v++) {
            if (m->use_hadamard) {
                int bin = ((int)X[i * m->n_vars + v] + m->bias[v]) & m->table_mask;
                hv_from_hadamard(pairs[v], v, bin);
            } else if (m->use_circulant && m->circulant.circulant_ready) {
                int bin = ((int)X[i * m->n_vars + v] + m->bias[v]) & m->table_mask;
                hv_from_circulant(pairs[v], &m->circulant, v, bin);
            } else {
                hv_t val_hv;
                an_hdc_encode(m, v, X[i * m->n_vars + v], val_hv);
                hv_bind(pairs[v], m->position_hvs[v], val_hv);
            }
        }
        /* Problem 6: Interaction encoding */
        for (int k = 0; k < m->n_interaction_pairs; k++) {
            int pi = m->interaction_pairs[k].i;
            int pj = m->interaction_pairs[k].j;
            hv_t hv_i, hv_j;
            an_hdc_encode(m, pi, X[i * m->n_vars + pi], hv_i);
            an_hdc_encode(m, pj, X[i * m->n_vars + pj], hv_j);
            hv_bind(pairs[m->n_vars + k], hv_i, hv_j);
        }
        hv_bundle(sample_hvs[i], pairs, total_pairs);
    }
    free(pairs);

    /* Iterative error correction */
    int *y_pred = (int *)malloc(n_samples * sizeof(int));
    if (!y_pred) { free(sample_hvs); if (epochs_run_out) *epochs_run_out = 0; return 0; }

    /* Track per-class bit counts for codebook modification */
    int16_t *cb_counts = (int16_t *)calloc(
        (size_t)m->n_classes * HDC_DIM, sizeof(int16_t));
    if (!cb_counts) { free(y_pred); free(sample_hvs); if (epochs_run_out) *epochs_run_out = 0; return 0; }

    /* Binary codebook for fast Hamming distance computation */
    hv_t *cb_binary = (hv_t *)aligned_alloc(64, m->n_classes * sizeof(hv_t));
    if (!cb_binary) { free(cb_counts); free(y_pred); free(sample_hvs);
                      if (epochs_run_out) *epochs_run_out = 0; return 0; }

    /* Initialize counts from current codebook with large weight to preserve initial training */
    for (int c = 0; c < m->n_classes; c++) {
        for (int bit = 0; bit < HDC_DIM; bit++) {
            int w = bit / 64;
            int b = bit % 64;
            if (m->codebook[c][w] & (1ULL << b))
                cb_counts[(size_t)c * HDC_DIM + bit] = 1000;
            else
                cb_counts[(size_t)c * HDC_DIM + bit] = -1000;
        }
        hv_copy(cb_binary[c], m->codebook[c]);
    }

    /* NeuralHD: precompute variance workspace */
    int *variance = NULL;
    if (regen_rate > 0.0f) {
        variance = (int *)malloc(HDC_DIM * sizeof(int));
        if (!variance) regen_rate = 0.0f;
    }

    /* ---- Problema 1: Early stopping state (monitors VALIDATION accuracy) ---- */
    double best_val_acc = 0.0;
    double best_train_acc = 0.0;
    int best_epoch = 0;
    int epochs_run = 0;
    int no_improve = 0;

    /* Save best codebook state */
    hv_t *cb_best = (hv_t *)aligned_alloc(64, m->n_classes * sizeof(hv_t));
    if (!cb_best) {
        free(variance); free(cb_binary); free(cb_counts); free(y_pred); free(sample_hvs);
        if (epochs_run_out) *epochs_run_out = 0;
        return 0;
    }
    for (int c = 0; c < m->n_classes; c++)
        hv_copy(cb_best[c], m->codebook[c]);

    for (int it = 0; it < n_iter; it++) {
        /* Predict on training split only [0, n_train) */
        for (int i = 0; i < n_train; i++)
            y_pred[i] = an_hdc_predict(m, &X[i * m->n_vars]);

        int n_updates = 0;
        int16_t lr_int = (int16_t)(lr * 2.0f);

        for (int i = 0; i < n_train; i++) {
            int pred = y_pred[i];
            int true_c = y[i];
            if (true_c < 0 || true_c >= m->n_classes) continue;
            if (pred < 0 || pred >= m->n_classes) continue;

            /* Skip if correct and no margin (AdaptHD pure behavior) */
            if (pred == true_c && margin <= 0) continue;

            int do_update = 0;
            int penalize_c = pred;

            if (pred != true_c) {
                do_update = 1;
                penalize_c = pred;
            } else if (margin > 0) {
                /* RefineHD: update if competitor within additive margin of true */
                int dist_true = hv_hamming_unrolled(sample_hvs[i], cb_binary[true_c]);
                int dist_comp = HDC_DIM + 1;
                int closest_comp = -1;
                for (int c = 0; c < m->n_classes; c++) {
                    if (c == true_c) continue;
                    int d = hv_hamming_unrolled(sample_hvs[i], cb_binary[c]);
                    if (d < dist_comp) { dist_comp = d; closest_comp = c; }
                }
                if (closest_comp >= 0 && dist_true + margin > dist_comp) {
                    do_update = 1;
                    penalize_c = closest_comp;
                }
            }

            if (!do_update) continue;
            n_updates++;

            for (int w = 0; w < HDC_WORDS; w++) {
                uint64_t word = sample_hvs[i][w];
                int base = w * 64;
                while (word) {
                    int bit = __builtin_ctzll(word);
                    int idx = base + bit;
                    cb_counts[(size_t)true_c * HDC_DIM + idx] += lr_int;
                    cb_counts[(size_t)penalize_c * HDC_DIM + idx] -= lr_int;
                    word &= word - 1;
                }
            }

            for (int w = 0; w < HDC_WORDS; w++) {
                uint64_t word = sample_hvs[i][w];
                int base = w * 64;
                while (word) {
                    int bit = __builtin_ctzll(word);
                    int idx = base + bit;
                    int64_t pos = (size_t)true_c * HDC_DIM + idx;
                    if (cb_counts[pos] > 0)
                        cb_binary[true_c][w] |= (1ULL << bit);
                    else
                        cb_binary[true_c][w] &= ~(1ULL << bit);
                    pos = (size_t)penalize_c * HDC_DIM + idx;
                    if (cb_counts[pos] > 0)
                        cb_binary[penalize_c][w] |= (1ULL << bit);
                    else
                        cb_binary[penalize_c][w] &= ~(1ULL << bit);
                    word &= word - 1;
                }
            }
        }

        if (n_updates == 0) break;

        /* Rebuild codebook from counts */
        for (int c = 0; c < m->n_classes; c++) {
            hv_zero(m->codebook[c]);
            for (int i = 0; i < HDC_DIM; i++) {
                if (cb_counts[(size_t)c * HDC_DIM + i] > 0)
                    m->codebook[c][i / 64] |= (1ULL << (i % 64));
            }
            hv_copy(cb_binary[c], m->codebook[c]);
        }

        /* NeuralHD: dimension regeneration (skip during warmup — first 5 epochs) */
        if (regen_rate > 0.0f && variance && it >= 5) {
            memset(variance, 0, HDC_DIM * sizeof(int));
            for (int d = 0; d < HDC_DIM; d++) {
                int count_ones = 0;
                int w = d / 64;
                int b = d % 64;
                for (int c = 0; c < m->n_classes; c++) {
                    if (m->codebook[c][w] & (1ULL << b))
                        count_ones++;
                }
                variance[d] = count_ones * (m->n_classes - count_ones);
            }

            int n_regen = (int)(regen_rate * (float)HDC_DIM);
            if (n_regen < 1) n_regen = 1;

            for (int r = 0; r < n_regen; r++) {
                int min_var = HDC_DIM * HDC_DIM;
                int min_idx = -1;
                for (int dd = 0; dd < HDC_DIM; dd++) {
                    if (variance[dd] < 0) continue;
                    if (variance[dd] < min_var) { min_var = variance[dd]; min_idx = dd; }
                }
                if (min_idx < 0) break;
                variance[min_idx] = -1;

                int w = min_idx / 64;
                int b = min_idx % 64;
                for (int c = 0; c < m->n_classes; c++) {
                    m->codebook[c][w] &= ~(1ULL << b);
                    cb_binary[c][w] &= ~(1ULL << b);
                    cb_counts[(size_t)c * HDC_DIM + min_idx] = 0;
                }

                uint64_t regen_seed = (uint64_t)it * 7919ULL + (uint64_t)min_idx * 104729ULL;
                for (int c = 0; c < m->n_classes; c++) {
                    regen_seed ^= (uint64_t)c * 6364136223846793005ULL;
                    regen_seed ^= regen_seed << 13;
                    regen_seed ^= regen_seed >> 7;
                    regen_seed ^= regen_seed << 17;
                    int bit_val = (int)(regen_seed & 1);
                    if (bit_val) {
                        m->codebook[c][w] |= (1ULL << b);
                        cb_binary[c][w] |= (1ULL << b);
                        cb_counts[(size_t)c * HDC_DIM + min_idx] = 50;
                    } else {
                        cb_counts[(size_t)c * HDC_DIM + min_idx] = -50;
                    }
                }

                /* Re-encode samples for regenerated dimension */
                {
                    hv_t *pairs_regen = (hv_t *)aligned_alloc(64,
                        total_pairs * sizeof(hv_t));
                    if (pairs_regen) {
                        for (int s = 0; s < n_samples; s++) {
                            for (int v = 0; v < m->n_vars; v++) {
                                if (m->use_hadamard) {
                                    int bin = ((int)X[s * m->n_vars + v] + m->bias[v]) & m->table_mask;
                                    hv_from_hadamard(pairs_regen[v], v, bin);
                                } else {
                                    hv_t val_hv;
                                    an_hdc_encode(m, v, X[s * m->n_vars + v], val_hv);
                                    hv_bind(pairs_regen[v], m->position_hvs[v], val_hv);
                                }
                            }
                            int bit_count = 0;
                            for (int v = 0; v < m->n_vars; v++)
                                if (pairs_regen[v][w] & (1ULL << b))
                                    bit_count++;
                            if (bit_count > m->n_vars / 2)
                                sample_hvs[s][w] |= (1ULL << b);
                            else
                                sample_hvs[s][w] &= ~(1ULL << b);
                        }
                        free(pairs_regen);
                    }
                }
            }
        }

        epochs_run = it + 1;

        /* ---- Problema 1: VALIDATION accuracy for early stopping ---- */
        int n_val_correct = 0;
        for (int i = n_train; i < n_samples; i++) {
            if (an_hdc_predict(m, &X[i * m->n_vars]) == y[i])
                n_val_correct++;
        }
        double val_acc = (n_val > 0) ? (double)n_val_correct / (double)n_val : 0.0;

        /* Training accuracy (for debug only) */
        int n_train_correct = 0;
        for (int i = 0; i < n_train; i++) {
            if (y_pred[i] == y[i]) n_train_correct++;
        }
        double train_acc = (n_train > 0) ? (double)n_train_correct / (double)n_train : 0.0;

        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            best_train_acc = train_acc;
            best_epoch = epochs_run;
            no_improve = 0;
            /* Save best codebook state */
            for (int c = 0; c < m->n_classes; c++)
                hv_copy(cb_best[c], m->codebook[c]);
        } else {
            no_improve++;
        }

        /* Verbose output */
        if (verbose > 0 && (epochs_run % verbose == 0 || epochs_run == 1)) {
            fprintf(stderr, "Epoch %4d/%d | val=%.1f%% | train=%.1f%% | best=%.1f%% @%d | patience=%d/%d\n",
                    epochs_run, n_iter,
                    val_acc * 100.0, train_acc * 100.0,
                    best_val_acc * 100.0, best_epoch,
                    no_improve, patience > 0 ? patience : 0);
        }

        /* Early stopping on VALIDATION accuracy */
        if (patience > 0 && no_improve >= patience) {
            if (verbose > 0)
                fprintf(stderr, "Early stop no epoch %d — melhor val accuracy: %.1f%%\n",
                        best_epoch, best_val_acc * 100.0);
            break;
        }
    }

    /* Restore best codebook (from validation) */
    for (int c = 0; c < m->n_classes; c++)
        hv_copy(m->codebook[c], cb_best[c]);

    free(cb_best);
    free(variance);
    free(cb_binary);
    free(cb_counts);
    free(y_pred);
    free(sample_hvs);

    if (epochs_run_out) *epochs_run_out = epochs_run;
    return epochs_run;
}

int an_hdc_predict(const an_hdc_model *m, const double *x) {
    if (!m || !x) return -1;

    /*
     * Inline encoding: for each variable, generate value HV via hv_from_seed,
     * bind with position HV, and accumulate into counters directly.
     * This avoids allocating and bundling separate hv_t arrays.
     * Problem 6: Also accumulates interaction pair counts.
     */
    uint16_t counts[HDC_DIM] = {0};
    int total_signals = m->n_vars + m->n_interaction_pairs;

    /* Cache per-variable encoded HVs for interaction reuse */
    hv_t *var_hvs = NULL;
    if (m->n_interaction_pairs > 0) {
        var_hvs = (hv_t *)aligned_alloc(64, m->n_vars * sizeof(hv_t));
    }

    /* Encode each variable: bind(pos, value) and accumulate bit counts */
    for (int v = 0; v < m->n_vars; v++) {
        int bin = ((int)x[v] + m->bias[v]) & m->table_mask;
        
        hv_t pair_hv;
        if (m->use_hadamard) {
            hv_from_hadamard(pair_hv, v, bin);
        } else if (m->use_circulant && m->circulant.circulant_ready) {
            hv_from_circulant(pair_hv, &m->circulant, v, bin);
        } else {
            hv_t val_hv;
            uint64_t seed = (uint64_t)v * 100003ULL + (uint64_t)bin;
            hv_from_seed(val_hv, seed);
            hv_bind(pair_hv, m->position_hvs[v], val_hv);
        }

        /* Cache for interaction encoding */
        if (var_hvs) hv_copy(var_hvs[v], pair_hv);

        /* Accumulate bit counts for bundling (majority vote) */
        for (int w = 0; w < HDC_WORDS; w++) {
            uint64_t word = pair_hv[w];
            int base = w * 64;
            while (word) {
                int bit = __builtin_ctzll(word);
                counts[base + bit]++;
                word &= word - 1;
            }
        }
    }

    /* Problem 6: Interaction pair encoding */
    if (var_hvs) {
        for (int k = 0; k < m->n_interaction_pairs; k++) {
            int pi = m->interaction_pairs[k].i;
            int pj = m->interaction_pairs[k].j;
            hv_t interaction_hv;
            hv_bind(interaction_hv, var_hvs[pi], var_hvs[pj]);
            for (int w = 0; w < HDC_WORDS; w++) {
                uint64_t word = interaction_hv[w];
                int base = w * 64;
                while (word) {
                    int bit = __builtin_ctzll(word);
                    counts[base + bit]++;
                    word &= word - 1;
                }
            }
        }
        free(var_hvs);
    }

    /* Build query HV from counters (majority vote threshold) */
    int threshold = total_signals / 2;
    hv_t query;
    memset(query, 0, sizeof(hv_t));
    for (int i = 0; i < HDC_DIM; i++) {
        if (counts[i] > threshold)
            query[i / 64] |= (1ULL << (i % 64));
    }

    /* Find closest class by Hamming distance (Solution 2A: early exit) */
    if (m->use_lsh && m->lsh_index) {
        int candidates[HDC_DIM];
        int n_cand = hdc_lsh_query(m->lsh_index, query, candidates, m->n_classes);
        
        if (n_cand > 0) {
            int best_idx = candidates[0];
            int best_d = hv_hamming_unrolled(query, m->codebook[best_idx]);
            for (int i = 1; i < n_cand; i++) {
                int d = hv_hamming_early_exit(query, m->codebook[candidates[i]], best_d);
                if (d < best_d) {
                    best_d = d;
                    best_idx = candidates[i];
                }
            }
            return best_idx;
        }
    }

    int best_c = 0;
    int best_d = hv_hamming_early_exit(query, m->codebook[0], HDC_DIM);
    for (int c = 1; c < m->n_classes; c++) {
        int d = hv_hamming_early_exit(query, m->codebook[c], best_d);
        if (d < best_d) {
            best_d = d;
            best_c = c;
        }
    }
    return best_c;
}

int an_hdc_predict_batch(const an_hdc_model *m, const double *X,
                         int *outputs, int n)
{
    if (!m || !X || !outputs || n <= 0) return -1;
    for (int i = 0; i < n; i++)
        outputs[i] = an_hdc_predict(m, &X[i * m->n_vars]);
    return 0;
}

/* ---- Melhoria 4: AVX2 batch prediction ---- */
int an_hdc_predict_batch_avx(const an_hdc_model *m, const double *X,
                              int *outputs, int n)
{
    if (!m || !X || !outputs || n <= 0) return -1;

    /* Process in groups of 4, remainder handled scalar */
    int i = 0;
    for (; i + 3 < n; i += 4) {
        /* Encode 4 samples: accumulate bit counts for each */
        uint16_t counts0[HDC_DIM] = {0};
        uint16_t counts1[HDC_DIM] = {0};
        uint16_t counts2[HDC_DIM] = {0};
        uint16_t counts3[HDC_DIM] = {0};

        for (int v = 0; v < m->n_vars; v++) {
            int bin0 = ((int)X[(i+0) * m->n_vars + v] + m->bias[v]) & m->table_mask;
            int bin1 = ((int)X[(i+1) * m->n_vars + v] + m->bias[v]) & m->table_mask;
            int bin2 = ((int)X[(i+2) * m->n_vars + v] + m->bias[v]) & m->table_mask;
            int bin3 = ((int)X[(i+3) * m->n_vars + v] + m->bias[v]) & m->table_mask;

            hv_t pair0, pair1, pair2, pair3;

            if (m->use_hadamard) {
                hv_from_hadamard(pair0, v, bin0);
                hv_from_hadamard(pair1, v, bin1);
                hv_from_hadamard(pair2, v, bin2);
                hv_from_hadamard(pair3, v, bin3);
            } else if (m->use_circulant && m->circulant.circulant_ready) {
                hv_from_circulant(pair0, &m->circulant, v, bin0);
                hv_from_circulant(pair1, &m->circulant, v, bin1);
                hv_from_circulant(pair2, &m->circulant, v, bin2);
                hv_from_circulant(pair3, &m->circulant, v, bin3);
            } else {
                hv_t val0, val1, val2, val3;
                hv_from_seed(val0, (uint64_t)v * 100003ULL + (uint64_t)bin0);
                hv_from_seed(val1, (uint64_t)v * 100003ULL + (uint64_t)bin1);
                hv_from_seed(val2, (uint64_t)v * 100003ULL + (uint64_t)bin2);
                hv_from_seed(val3, (uint64_t)v * 100003ULL + (uint64_t)bin3);
                hv_bind(pair0, m->position_hvs[v], val0);
                hv_bind(pair1, m->position_hvs[v], val1);
                hv_bind(pair2, m->position_hvs[v], val2);
                hv_bind(pair3, m->position_hvs[v], val3);
            }

            /* Accumulate bit counts for all 4 samples */
            for (int w = 0; w < HDC_WORDS; w++) {
                int base = w * 64;

                uint64_t word0 = pair0[w];
                while (word0) {
                    int bit = __builtin_ctzll(word0);
                    counts0[base + bit]++;
                    word0 &= word0 - 1;
                }

                uint64_t word1 = pair1[w];
                while (word1) {
                    int bit = __builtin_ctzll(word1);
                    counts1[base + bit]++;
                    word1 &= word1 - 1;
                }

                uint64_t word2 = pair2[w];
                while (word2) {
                    int bit = __builtin_ctzll(word2);
                    counts2[base + bit]++;
                    word2 &= word2 - 1;
                }

                uint64_t word3 = pair3[w];
                while (word3) {
                    int bit = __builtin_ctzll(word3);
                    counts3[base + bit]++;
                    word3 &= word3 - 1;
                }
            }
        }

        /* Build query HVs from counts */
        hv_t query0, query1, query2, query3;
        memset(query0, 0, sizeof(hv_t));
        memset(query1, 0, sizeof(hv_t));
        memset(query2, 0, sizeof(hv_t));
        memset(query3, 0, sizeof(hv_t));

        int threshold = m->n_vars / 2;
        for (int d = 0; d < HDC_DIM; d++) {
            int w = d / 64;
            uint64_t mask = 1ULL << (d % 64);
            if (counts0[d] > threshold) query0[w] |= mask;
            if (counts1[d] > threshold) query1[w] |= mask;
            if (counts2[d] > threshold) query2[w] |= mask;
            if (counts3[d] > threshold) query3[w] |= mask;
        }

        /* Batch Hamming distance: 4 queries vs each codebook entry */
        hv_t queries[4] = {0};
        memcpy(queries[0], query0, sizeof(hv_t));
        memcpy(queries[1], query1, sizeof(hv_t));
        memcpy(queries[2], query2, sizeof(hv_t));
        memcpy(queries[3], query3, sizeof(hv_t));

        int best_c0 = 0, best_d0 = HDC_DIM + 1;
        int best_c1 = 0, best_d1 = HDC_DIM + 1;
        int best_c2 = 0, best_d2 = HDC_DIM + 1;
        int best_c3 = 0, best_d3 = HDC_DIM + 1;

        for (int c = 0; c < m->n_classes; c++) {
            int dists[4] = {0, 0, 0, 0};
            hv_hamming_batch4(queries, m->codebook[c], dists);
            if (dists[0] < best_d0) { best_d0 = dists[0]; best_c0 = c; }
            if (dists[1] < best_d1) { best_d1 = dists[1]; best_c1 = c; }
            if (dists[2] < best_d2) { best_d2 = dists[2]; best_c2 = c; }
            if (dists[3] < best_d3) { best_d3 = dists[3]; best_c3 = c; }
        }

        outputs[i+0] = best_c0;
        outputs[i+1] = best_c1;
        outputs[i+2] = best_c2;
        outputs[i+3] = best_c3;
    }

    /* Handle remainder scalar */
    for (; i < n; i++)
        outputs[i] = an_hdc_predict(m, &X[i * m->n_vars]);

    return 0;
}

/* ---- OPT-1: Cache functions ---- */

void an_hdc_warm_cache(an_hdc_model *m) {
    if (!m) return;
    if (m->cache) hdc_cache_free(m->cache);
    m->cache = hdc_cache_create(m->n_vars, m->table_size);
    if (m->cache) {
        hdc_cache_build(m->cache);
        m->use_cache = 1;
    }
}

void an_hdc_set_cache(an_hdc_model *m, int use_cache) {
    if (!m) return;
    if (use_cache && !m->cache) {
        an_hdc_warm_cache(m);
    } else if (!use_cache) {
        m->use_cache = 0;
    } else {
        m->use_cache = 1;
    }
}

/* ---- OPT-5: Multithreaded batch prediction ---- */

typedef struct {
    const an_hdc_model *model;
    const double *X;
    int *y_pred;
    int start;
    int end;
} predict_thread_args;

static void *predict_worker(void *arg) {
    predict_thread_args *a = (predict_thread_args *)arg;
    for (int i = a->start; i < a->end; i++)
        a->y_pred[i] = an_hdc_predict(a->model, &a->X[i * a->model->n_vars]);
    return NULL;
}

int an_hdc_predict_batch_mt(const an_hdc_model *m, const double *X,
                            int *outputs, int n, int n_threads)
{
    if (!m || !X || !outputs || n <= 0) return -1;

    if (n_threads <= 0) {
        n_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (n_threads <= 0) n_threads = 1;
    }
    if (n_threads > n) n_threads = n;

    pthread_t threads[64];
    predict_thread_args args[64];
    int chunk = n / n_threads;

    for (int t = 0; t < n_threads; t++) {
        args[t] = (predict_thread_args){
            .model  = m,
            .X      = X,
            .y_pred = outputs,
            .start  = t * chunk,
            .end    = (t == n_threads - 1) ? n : (t + 1) * chunk
        };
        pthread_create(&threads[t], NULL, predict_worker, &args[t]);
    }
    for (int t = 0; t < n_threads; t++)
        pthread_join(threads[t], NULL);
    return 0;
}

void an_hdc_set_threads(an_hdc_model *m, int n_threads) {
    if (!m) return;
    m->n_threads = n_threads;
}

/* ---- Save / Load ---- */

int an_hdc_save(const an_hdc_model *m, const char *path) {
    if (!m || !path) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Header */
    fwrite(&m->n_vars,     sizeof(int), 1, f);
    fwrite(&m->n_classes,  sizeof(int), 1, f);
    fwrite(&m->table_size, sizeof(int), 1, f);

    /* Bias */
    fwrite(m->bias, sizeof(int), m->n_vars, f);

    /* Position hypervectors */
    fwrite(m->position_hvs, sizeof(hv_t), m->n_vars, f);

    /* Codebook */
    fwrite(m->codebook, sizeof(hv_t), m->n_classes, f);

    fclose(f);
    return 0;
}

an_hdc_model *an_hdc_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    int n_vars, n_classes, table_size;
    if (fread(&n_vars,     sizeof(int), 1, f) != 1) goto fail;
    if (fread(&n_classes,  sizeof(int), 1, f) != 1) goto fail;
    if (fread(&table_size, sizeof(int), 1, f) != 1) goto fail;

    an_hdc_model *m = an_hdc_create(n_vars, n_classes, table_size, NULL);
    if (!m) goto fail;

    /* Re-read bias (overwrite auto-generated) */
    if (fread(m->bias, sizeof(int), n_vars, f) != (size_t)n_vars) {
        an_hdc_free(m); goto fail;
    }

    /* Read position hypervectors */
    if (fread(m->position_hvs, sizeof(hv_t), n_vars, f) != (size_t)n_vars) {
        an_hdc_free(m); goto fail;
    }

    /* Read codebook */
    if (fread(m->codebook, sizeof(hv_t), n_classes, f) != (size_t)n_classes) {
        an_hdc_free(m); goto fail;
    }

    fclose(f);
    return m;

fail:
    fclose(f);
    return NULL;
}

/* PROMPT-1: Circulant encoding control */
void an_hdc_set_circulant(an_hdc_model *m, int enable) {
    if (!m) return;
    if (enable && !m->circulant.circulant_ready) {
        hdc_circulant_init(&m->circulant, 0xdeadbeef12345678ULL, 0xcafe9876abcd0123ULL);
    }
    m->use_circulant = enable;
}

/* Melhoria 3: Hadamard encoding control */
void an_hdc_set_hadamard(an_hdc_model *m, int enable) {
    if (!m) return;
    m->use_hadamard = enable;
}

/* PROMPT-2: Early termination control */
void an_hdc_set_early_termination(an_hdc_model *m, int enable, int margin) {
    if (!m) return;
    m->use_early_term = enable;
    if (margin > 0) {
        m->early_term_margin = margin;
    }
}

/* PROMPT-3: CompHD dimension folding */
void an_hdc_fold_codebook(an_hdc_model *m, int fold) {
    if (!m || fold <= 0) return;
    
    if (m->codebook_folded) free(m->codebook_folded);
    
    m->fold_factor = fold;
    m->codebook_folded = (hv_folded_t *)malloc(m->n_classes * sizeof(hv_folded_t));
    if (!m->codebook_folded) return;
    
    for (int c = 0; c < m->n_classes; c++) {
        hv_fold(m->codebook_folded[c], m->codebook[c], fold);
    }
    
    m->use_folded = 1;
}

int an_hdc_predict_folded(const an_hdc_model *m, const double *x) {
    if (!m || !x || !m->codebook_folded) return -1;

    uint16_t counts[HDC_DIM] = {0};

    for (int v = 0; v < m->n_vars; v++) {
        int bin = ((int)x[v] + m->bias[v]) & m->table_mask;
        hv_t val_hv;

        if (m->use_hadamard) {
            hv_from_hadamard(val_hv, v, bin);
        } else {
            uint64_t seed = (uint64_t)v * 100003ULL + (uint64_t)bin;
            hv_from_seed(val_hv, seed);
        }

        for (int w = 0; w < HDC_WORDS; w++) {
            uint64_t word;
            if (m->use_hadamard) {
                word = val_hv[w];
            } else {
                word = m->position_hvs[v][w] ^ val_hv[w];
            }
            int base = w * 64;
            while (word) {
                int bit = __builtin_ctzll(word);
                counts[base + bit]++;
                word &= word - 1;
            }
        }
    }

    int threshold = m->n_vars / 2;
    hv_t query;
    memset(query, 0, sizeof(hv_t));
    for (int i = 0; i < HDC_DIM; i++) {
        if (counts[i] > threshold)
            query[i / 64] |= (1ULL << (i % 64));
    }

    hv_folded_t query_folded;
    hv_fold(query_folded, query, m->fold_factor);

    int best_c = 0;
    int best_d = hv_hamming_folded(query_folded, m->codebook_folded[0]);
    for (int c = 1; c < m->n_classes; c++) {
        int d = hv_hamming_folded(query_folded, m->codebook_folded[c]);
        if (d < best_d) {
            best_d = d;
            best_c = c;
        }
    }
    return best_c;
}

/* LSH: Locality-Sensitive Hashing */
void an_hdc_build_lsh(an_hdc_model *m) {
    if (!m) return;
    if (m->lsh_index) hdc_lsh_free(m->lsh_index);
    m->lsh_index = hdc_lsh_build(m->codebook, m->n_classes);
    m->use_lsh = 1;
}

void an_hdc_build_lsh_ex(an_hdc_model *m, int k, int l) {
    if (!m) return;
    if (m->lsh_index) hdc_lsh_free(m->lsh_index);
    m->lsh_index = hdc_lsh_build_ex(m->codebook, m->n_classes, k, l);
    m->use_lsh = 1;
}

void an_hdc_set_lsh(an_hdc_model *m, int enable) {
    if (!m) return;
    m->use_lsh = enable;
}

/* predict_top_k: Retorna os K classes mais próximas (menor distância Hamming) */
void an_hdc_predict_top_k(const an_hdc_model *m, const double *x,
                          int *out_classes, int k) {
    if (!m || !x || !out_classes || k <= 0) return;
    if (k > m->n_classes) k = m->n_classes;

    /* Encode input to query hypervector (same logic as an_hdc_predict) */
    uint16_t counts[HDC_DIM] = {0};

    for (int v = 0; v < m->n_vars; v++) {
        int bin = ((int)x[v] + m->bias[v]) & m->table_mask;
        
        hv_t pair_hv;
        if (m->use_hadamard) {
            hv_from_hadamard(pair_hv, v, bin);
        } else if (m->use_circulant && m->circulant.circulant_ready) {
            hv_from_circulant(pair_hv, &m->circulant, v, bin);
        } else {
            hv_t val_hv;
            uint64_t seed = (uint64_t)v * 100003ULL + (uint64_t)bin;
            hv_from_seed(val_hv, seed);
            hv_bind(pair_hv, m->position_hvs[v], val_hv);
        }

        for (int w = 0; w < HDC_WORDS; w++) {
            uint64_t word = pair_hv[w];
            int base = w * 64;
            while (word) {
                int bit = __builtin_ctzll(word);
                counts[base + bit]++;
                word &= word - 1;
            }
        }
    }

    int threshold = m->n_vars / 2;
    hv_t query;
    memset(query, 0, sizeof(hv_t));
    for (int i = 0; i < HDC_DIM; i++) {
        if (counts[i] > threshold)
            query[i / 64] |= (1ULL << (i % 64));
    }

    /* Calculate distances to all classes and find top K */
    int *dists = (int *)malloc(m->n_classes * sizeof(int));
    if (!dists) return;

    for (int c = 0; c < m->n_classes; c++) {
        dists[c] = hv_hamming_unrolled(query, m->codebook[c]);
    }

    /* Selection sort for top K (K is small, typically <= 10) */
    for (int i = 0; i < k; i++) {
        int min_c = i;
        for (int c = i + 1; c < m->n_classes; c++) {
            if (dists[c] < dists[min_c]) min_c = c;
        }
        out_classes[i] = min_c;
        dists[min_c] = INT_MAX;
    }

    free(dists);
}

/* Problem 6: Set interaction pairs for feature interaction encoding */
void an_hdc_set_interactions(an_hdc_model *m, const int *pairs_i,
                              const int *pairs_j, int n_pairs) {
    if (!m) return;
    if (m->interaction_pairs) { free(m->interaction_pairs); m->interaction_pairs = NULL; }
    m->n_interaction_pairs = 0;

    if (n_pairs <= 0 || !pairs_i || !pairs_j) return;

    m->interaction_pairs = (an_interaction_pair_t *)malloc(
        n_pairs * sizeof(an_interaction_pair_t));
    if (!m->interaction_pairs) return;

    for (int k = 0; k < n_pairs; k++) {
        m->interaction_pairs[k].i = pairs_i[k];
        m->interaction_pairs[k].j = pairs_j[k];
    }
    m->n_interaction_pairs = n_pairs;
}
