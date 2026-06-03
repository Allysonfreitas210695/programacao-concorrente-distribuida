#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Uso: %s <num_elementos> <num_buckets> <num_threads>\n", argv[0]);
        return 1;
    }

    int n            = atoi(argv[1]);
    int num_buckets  = atoi(argv[2]);
    int thread_count = atoi(argv[3]);

    int* dados = (int*)malloc(n * sizeof(int));
    srand(42);
    for (int i = 0; i < n; i++)
        dados[i] = rand() % num_buckets;

    int* hist = (int*)calloc(num_buckets, sizeof(int));

    double inicio = omp_get_wtime();

    /*
     * Histogramas locais por thread para evitar condição de corrida
     * sem usar critical a cada incremento (melhor desempenho).
     */
    #pragma omp parallel num_threads(thread_count)
    {
        int* hist_local = (int*)calloc(num_buckets, sizeof(int));

        #pragma omp for
        for (int i = 0; i < n; i++) {
            hist_local[dados[i]]++;
        }

        #pragma omp critical
        {
            for (int b = 0; b < num_buckets; b++)
                hist[b] += hist_local[b];
        }

        free(hist_local);
    }

    double fim = omp_get_wtime();

    printf("Histograma (%d elementos, %d buckets, %d threads):\n",
           n, num_buckets, thread_count);
    for (int b = 0; b < num_buckets; b++) {
        printf("  Bucket [%3d]: %d\n", b, hist[b]);
    }
    printf("Tempo: %.4f segundos\n", fim - inicio);

    free(dados);
    free(hist);
    return 0;
}
