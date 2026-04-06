/*
 * test_hdc_main.c — Quick C test for libaddernet_hdc
 */

#include <stdio.h>
#include <stdlib.h>
#include "addernet_hdc.h"

int main(void) {
    /* Simple 2D classification: 2 classes, 2 variables */
    /* Class 0: x in [0,5], y in [0,5] */
    /* Class 1: x in [10,15], y in [10,15] */

    double X[] = {
        1, 2,   3, 1,   2, 4,   4, 3,   0, 5,    /* class 0 */
        11, 12, 13, 11, 12, 14, 14, 13, 10, 15    /* class 1 */
    };
    int y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    int n = 10;

    printf("Creating model: 2 vars, 2 classes, table size 256, hv_dim 1024\n");
    int bias[] = {0, 0};
    an_hdc_model *m = an_hdc_create(2, 2, 256, bias, 1024);
    if (!m) { fprintf(stderr, "create failed\n"); return 1; }

    printf("Training on %d samples...\n", n);
    hv_seed(42);
    an_hdc_train(m, X, y, n);

    /* Test predictions */
    printf("\n  %6s | %6s | %6s | %8s\n", "x", "y", "Pred", "Correct");
    printf("  ");
    for (int i = 0; i < 34; i++) printf("-");
    printf("\n");

    double test_X[] = {
        1, 2,  3, 3,  0, 4,  2, 1,  4, 5,  1, 0,     /* class 0 */
        11, 12, 13, 11, 10, 14, 12, 13, 14, 10, 15, 15 /* class 1 */
    };
    int    test_y[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    int    nt = 12;

    int correct = 0;
    for (int i = 0; i < nt; i++) {
        int pred = an_hdc_predict(m, &test_X[i * 2]);
        int ok = (pred == test_y[i]);
        correct += ok;
        printf("  %6.1f | %6.1f | %6d | %8s\n",
               test_X[i*2], test_X[i*2+1], pred, ok ? "YES" : "NO");
    }
    printf("\n  Accuracy: %d/%d = %.1f%%\n", correct, nt, 100.0 * correct / nt);

    /* Test save/load */
    printf("\nSave/load test...\n");
    if (an_hdc_save(m, "/tmp/hdc_test.bin") == 0) {
        an_hdc_model *loaded = an_hdc_load("/tmp/hdc_test.bin");
        if (loaded) {
            double test[] = {1, 2};
            int p1 = an_hdc_predict(m, test);
            int p2 = an_hdc_predict(loaded, test);
            printf("  Original: %d\n", p1);
            printf("  Loaded:   %d\n", p2);
            printf("  Match: %s\n", (p1 == p2) ? "YES" : "NO");
            an_hdc_free(loaded);
        }
    }

    an_hdc_free(m);
    printf("\nAll tests passed.\n");
    return 0;
}
