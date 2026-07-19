#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "addernet.h"

typedef struct { int c; double f; } an_sample;

static int sample_compare(const void *a, const void *b) {
    const an_sample *sa = (const an_sample *)a;
    const an_sample *sb = (const an_sample *)b;
    return (sa->c > sb->c) - (sa->c < sb->c);
}

/* Sort, merge duplicate integer coordinates, and interpolate/extrapolate. */
static int an_expand(const an_sample *src, int ns, an_sample *out, int lo, int hi) {
    if (!src || !out || ns <= 0 || hi < lo) return 0;
    if ((size_t)ns > SIZE_MAX / sizeof(an_sample)) return 0;

    an_sample *sorted = (an_sample *)malloc((size_t)ns * sizeof(an_sample));
    an_sample *unique = (an_sample *)malloc((size_t)ns * sizeof(an_sample));
    if (!sorted || !unique) {
        free(sorted); free(unique); return 0;
    }
    memcpy(sorted, src, (size_t)ns * sizeof(an_sample));
    qsort(sorted, (size_t)ns, sizeof(an_sample), sample_compare);

    int nu = 0;
    for (int i = 0; i < ns;) {
        int coordinate = sorted[i].c;
        double sum = 0.0;
        int count = 0;
        while (i < ns && sorted[i].c == coordinate) {
            sum += sorted[i].f;
            count++;
            i++;
        }
        unique[nu].c = coordinate;
        unique[nu].f = sum / (double)count;
        nu++;
    }
    free(sorted);

    double left_slope = 0.0, right_slope = 0.0;
    if (nu > 1) {
        left_slope = (unique[1].f - unique[0].f) /
                     (double)(unique[1].c - unique[0].c);
        right_slope = (unique[nu - 1].f - unique[nu - 2].f) /
                      (double)(unique[nu - 1].c - unique[nu - 2].c);
    }

    int cursor = 0;
    int n = 0;
    for (int v = lo; v <= hi; v++) {
        out[n].c = v;
        if (nu == 1) {
            out[n].f = unique[0].f;
        } else if (v <= unique[0].c) {
            out[n].f = unique[0].f + left_slope * (double)(v - unique[0].c);
        } else if (v >= unique[nu - 1].c) {
            out[n].f = unique[nu - 1].f + right_slope * (double)(v - unique[nu - 1].c);
        } else {
            while (cursor + 1 < nu && unique[cursor + 1].c < v) cursor++;
            const an_sample a = unique[cursor];
            const an_sample b = unique[cursor + 1];
            const double frac = (double)(v - a.c) / (double)(b.c - a.c);
            out[n].f = a.f + frac * (b.f - a.f);
        }
        n++;
        if (v == INT_MAX) break;
    }
    free(unique);
    return n;
}

static void an_train_samples(an_layer *layer, const an_sample *data, int n, int epochs) {
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < n; i++) {
            int idx = (data[i].c + layer->bias) & layer->mask;
            double current = layer->offset[idx];
            double err = fabs(current - data[i].f);
            double up_err = fabs((current + layer->lr) - data[i].f);
            double down_err = fabs((current - layer->lr) - data[i].f);
            if (up_err < err && up_err <= down_err) layer->offset[idx] = current + layer->lr;
            else if (down_err < err) layer->offset[idx] = current - layer->lr;
        }
    }
}

an_layer *an_layer_create(int size, int bias, int input_min, int input_max, double lr) {
    if (size <= 0 || (size & (size - 1)) != 0 || size > (1 << 26)) return NULL;
    if (input_max < input_min || !isfinite(lr) || lr <= 0.0) return NULL;
    if ((size_t)size > SIZE_MAX / sizeof(double)) return NULL;

    an_layer *layer = (an_layer *)an_aligned_alloc(64, sizeof(an_layer));
    if (!layer) return NULL;
    layer->offset = (double *)an_aligned_alloc(64, (size_t)size * sizeof(double));
    if (!layer->offset) { an_aligned_free(layer); return NULL; }

    layer->size = size;
    layer->mask = size - 1;
    layer->bias = bias;
    layer->input_min = input_min;
    layer->input_max = input_max;
    layer->lr = lr;
    memset(layer->offset, 0, (size_t)size * sizeof(double));
    return layer;
}

void an_layer_free(an_layer *layer) {
    if (!layer) return;
    an_aligned_free(layer->offset);
    layer->offset = NULL;
    an_aligned_free(layer);
}

int an_train(an_layer *layer, const double *inputs, const double *targets,
             int n_samples, int epochs_raw, int epochs_expanded) {
    if (!layer || !inputs || !targets || n_samples <= 0 ||
        epochs_raw < 0 || epochs_expanded < 0) return -1;
    if ((size_t)n_samples > SIZE_MAX / sizeof(an_sample)) return -1;

    an_sample *raw = (an_sample *)malloc((size_t)n_samples * sizeof(an_sample));
    if (!raw) return -1;
    for (int i = 0; i < n_samples; i++) {
        if (!isfinite(inputs[i]) || !isfinite(targets[i]) ||
            inputs[i] < (double)INT_MIN || inputs[i] > (double)INT_MAX) {
            free(raw); return -2;
        }
        raw[i].c = (int)inputs[i];
        raw[i].f = targets[i];
    }

    an_train_samples(layer, raw, n_samples, epochs_raw);
    long long range = (long long)layer->input_max - (long long)layer->input_min + 1LL;
    if (epochs_expanded > 0 && range > 0 && range <= INT_MAX &&
        (size_t)range <= SIZE_MAX / sizeof(an_sample)) {
        an_sample *dense = (an_sample *)malloc((size_t)range * sizeof(an_sample));
        if (!dense) { free(raw); return -1; }
        int nd = an_expand(raw, n_samples, dense, layer->input_min, layer->input_max);
        if (nd <= 0) { free(dense); free(raw); return -1; }
        an_train_samples(layer, dense, nd, epochs_expanded);
        free(dense);
    }
    free(raw);
    return 0;
}

int an_fit_direct(an_layer *layer, const double *inputs, const double *targets,
                  int n_samples, int interpolate, double blend) {
    if (!layer || !inputs || !targets || n_samples <= 0 ||
        !isfinite(blend) || blend <= 0.0 || blend > 1.0) return -1;
    if ((size_t)n_samples > SIZE_MAX / sizeof(an_sample)) return -1;

    an_sample *raw = (an_sample *)malloc((size_t)n_samples * sizeof(an_sample));
    if (!raw) return -1;
    for (int i = 0; i < n_samples; i++) {
        if (!isfinite(inputs[i]) || !isfinite(targets[i]) ||
            inputs[i] < (double)INT_MIN || inputs[i] > (double)INT_MAX) {
            free(raw); return -2;
        }
        raw[i].c = (int)inputs[i];
        raw[i].f = targets[i];
    }

    double keep = 1.0 - blend;
    if (interpolate) {
        long long range = (long long)layer->input_max - (long long)layer->input_min + 1LL;
        if (range <= 0 || range > INT_MAX || (size_t)range > SIZE_MAX / sizeof(an_sample)) {
            free(raw); return -1;
        }
        an_sample *dense = (an_sample *)malloc((size_t)range * sizeof(an_sample));
        if (!dense) { free(raw); return -1; }
        int nd = an_expand(raw, n_samples, dense, layer->input_min, layer->input_max);
        if (nd <= 0) { free(dense); free(raw); return -1; }
        for (int i = 0; i < nd; i++) {
            int idx = (dense[i].c + layer->bias) & layer->mask;
            layer->offset[idx] = keep * layer->offset[idx] + blend * dense[i].f;
        }
        free(dense);
    } else {
        double *sums = (double *)calloc((size_t)layer->size, sizeof(double));
        uint64_t *counts = (uint64_t *)calloc((size_t)layer->size, sizeof(uint64_t));
        if (!sums || !counts) { free(sums); free(counts); free(raw); return -1; }
        for (int i = 0; i < n_samples; i++) {
            int idx = (raw[i].c + layer->bias) & layer->mask;
            sums[idx] += raw[i].f;
            counts[idx]++;
        }
        for (int i = 0; i < layer->size; i++) {
            if (counts[i]) {
                double mean = sums[i] / (double)counts[i];
                layer->offset[i] = keep * layer->offset[i] + blend * mean;
            }
        }
        free(sums); free(counts);
    }
    free(raw);
    return 0;
}

double an_predict(const an_layer *layer, double input) {
    if (!layer || !isfinite(input) || input < (double)INT_MIN || input > (double)INT_MAX) return NAN;
    int idx = ((int)input + layer->bias) & layer->mask;
    return layer->offset[idx];
}

int an_predict_batch(const an_layer *layer, const double *inputs, double *outputs, int n) {
    if (!layer || !inputs || !outputs || n <= 0) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++) {
        if (!isfinite(inputs[i]) || inputs[i] < (double)INT_MIN || inputs[i] > (double)INT_MAX) {
            outputs[i] = NAN;
        } else {
            int idx = ((int)inputs[i] + layer->bias) & layer->mask;
            outputs[i] = layer->offset[idx];
        }
    }
    return 0;
}

int an_save(const an_layer *layer, const char *path) {
    if (!layer || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ok = 1;
    ok &= fwrite(&layer->size, sizeof(int), 1, f) == 1;
    ok &= fwrite(&layer->bias, sizeof(int), 1, f) == 1;
    ok &= fwrite(&layer->input_min, sizeof(int), 1, f) == 1;
    ok &= fwrite(&layer->input_max, sizeof(int), 1, f) == 1;
    ok &= fwrite(&layer->lr, sizeof(double), 1, f) == 1;
    ok &= fwrite(layer->offset, sizeof(double), (size_t)layer->size, f) == (size_t)layer->size;
    if (fflush(f) != 0) ok = 0;
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

an_layer *an_load(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int size, bias, input_min, input_max;
    double lr;
    if (fread(&size, sizeof(int), 1, f) != 1 ||
        fread(&bias, sizeof(int), 1, f) != 1 ||
        fread(&input_min, sizeof(int), 1, f) != 1 ||
        fread(&input_max, sizeof(int), 1, f) != 1 ||
        fread(&lr, sizeof(double), 1, f) != 1) goto fail;
    if (size <= 0 || (size & (size - 1)) != 0 || size > (1 << 26) ||
        input_max < input_min || !isfinite(lr) || lr <= 0.0) goto fail;
    an_layer *layer = an_layer_create(size, bias, input_min, input_max, lr);
    if (!layer) goto fail;
    if (fread(layer->offset, sizeof(double), (size_t)size, f) != (size_t)size) {
        an_layer_free(layer); goto fail;
    }
    for (int i = 0; i < size; i++) {
        if (!isfinite(layer->offset[i])) { an_layer_free(layer); goto fail; }
    }
    fclose(f);
    return layer;
fail:
    fclose(f);
    return NULL;
}

int an_get_offset(const an_layer *layer, double *buf, int buf_size) {
    if (!layer || !buf || buf_size < layer->size) return -1;
    memcpy(buf, layer->offset, (size_t)layer->size * sizeof(double));
    return 0;
}

int an_set_offset(an_layer *layer, const double *buf, int buf_size) {
    if (!layer || !buf || buf_size != layer->size) return -1;
    for (int i = 0; i < buf_size; i++) if (!isfinite(buf[i])) return -2;
    memcpy(layer->offset, buf, (size_t)layer->size * sizeof(double));
    return 0;
}

int an_get_size(const an_layer *layer) { return layer ? layer->size : 0; }
int an_get_bias(const an_layer *layer) { return layer ? layer->bias : 0; }
int an_get_input_min(const an_layer *layer) { return layer ? layer->input_min : 0; }
int an_get_input_max(const an_layer *layer) { return layer ? layer->input_max : 0; }
double an_get_lr(const an_layer *layer) { return layer ? layer->lr : 0.0; }

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
void *an_layer_mmap_load(const char *path) {
    if (!path) return NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size <= 0) { close(fd); return NULL; }
    void *mapped = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return mapped == MAP_FAILED ? NULL : mapped;
}
#endif
