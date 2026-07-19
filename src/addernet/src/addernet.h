/*
 * AdderNet scalar lookup layer — public C API.
 * Inference is a table lookup indexed with addition and a bit mask.
 */
#ifndef ADDERNET_H
#define ADDERNET_H

#include <stdalign.h>
#include <stdlib.h>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline void *an_aligned_alloc(size_t alignment, size_t size) {
    size_t padded = (size + alignment - 1) & ~(alignment - 1);
#if defined(_MSC_VER)
    return _aligned_malloc(padded, alignment);
#else
    return aligned_alloc(alignment, padded);
#endif
}

static inline void an_aligned_free(void *ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

typedef struct {
    int size;
    int mask;
    int bias;
    int input_min;
    int input_max;
    double lr;
    double *offset;
} an_layer;

an_layer *an_layer_create(int size, int bias, int input_min, int input_max, double lr);
void an_layer_free(an_layer *layer);

int an_train(an_layer *layer, const double *inputs, const double *targets,
             int n_samples, int epochs_raw, int epochs_expanded);

/* Direct O(n + range) fitting. Duplicate integer inputs are averaged and
 * optional interpolation fills the configured input domain. blend=1 replaces
 * the table; values in (0, 1) blend with the previous table. */
int an_fit_direct(an_layer *layer, const double *inputs, const double *targets,
                  int n_samples, int interpolate, double blend);

double an_predict(const an_layer *layer, double input);
int an_predict_batch(const an_layer *layer, const double *inputs,
                     double *outputs, int n);

int an_save(const an_layer *layer, const char *path);
an_layer *an_load(const char *path);

int an_get_offset(const an_layer *layer, double *buf, int buf_size);
int an_set_offset(an_layer *layer, const double *buf, int buf_size);

int an_get_size(const an_layer *layer);
int an_get_bias(const an_layer *layer);
int an_get_input_min(const an_layer *layer);
int an_get_input_max(const an_layer *layer);
double an_get_lr(const an_layer *layer);

#ifdef __linux__
void *an_layer_mmap_load(const char *path);
#endif

#ifdef __cplusplus
}
#endif
#endif
