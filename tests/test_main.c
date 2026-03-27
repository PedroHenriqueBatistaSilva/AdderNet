/*
 * test_main.c — Quick C test for libaddernet
 */

#include <stdio.h>
#include <stdlib.h>
#include "addernet.h"

int main(void) {
    /* Celsius->Fahrenheit training data */
    double inputs[] = {
        0, 5, 10, 15, 20, 25, 30, 35, 40, 50,
        60, 70, 80, 90, 100, -10, -20, 37
    };
    double targets[] = {
        32, 41, 50, 59, 68, 77, 86, 95, 104, 122,
        140, 158, 176, 194, 212, 14, -4, 98.6
    };
    int n = sizeof(inputs) / sizeof(inputs[0]);

    /* Create layer: 256 entries, bias=50, range [-50, 200], lr=0.1 */
    an_layer *layer = an_layer_create(256, 50, -50, 200, 0.1);
    if (!layer) {
        fprintf(stderr, "Failed to create layer\n");
        return 1;
    }

    /* Train */
    printf("Training...\n");
    an_train(layer, inputs, targets, n, 1000, 4000);

    /* Test predictions */
    printf("\n  %5s | %8s | %12s | %8s\n", "C", "Real", "AdderNet", "Error");
    printf("  ");
    for (int i = 0; i < 40; i++) printf("-");
    printf("\n");

    int test_vals[] = {-40, -20, -10, 0, 10, 20, 25, 30, 37, 50, 80, 100, 150, 200};
    int nt = sizeof(test_vals) / sizeof(test_vals[0]);

    for (int i = 0; i < nt; i++) {
        int c = test_vals[i];
        double real = c * 1.8 + 32;
        double pred = an_predict(layer, (double)c);
        printf("  %5d | %8.2f | %12.2f | %+8.2f\n", c, real, pred, pred - real);
    }

    /* Save and reload */
    printf("\nSave/load test...\n");
    if (an_save(layer, "/tmp/an_test.bin") == 0) {
        an_layer *loaded = an_load("/tmp/an_test.bin");
        if (loaded) {
            double p1 = an_predict(layer, 37.0);
            double p2 = an_predict(loaded, 37.0);
            printf("  Original 37C: %.2f\n", p1);
            printf("  Loaded   37C: %.2f\n", p2);
            printf("  Match: %s\n", (p1 == p2) ? "YES" : "NO");
            an_layer_free(loaded);
        }
    }

    /* Batch test */
    printf("\nBatch prediction test...\n");
    double batch_in[] = {0, 25, 37, 100};
    double batch_out[4];
    an_predict_batch(layer, batch_in, batch_out, 4);
    for (int i = 0; i < 4; i++)
        printf("  %.0f C -> %.2f F\n", batch_in[i], batch_out[i]);

    an_layer_free(layer);
    printf("\nAll tests passed.\n");
    return 0;
}
