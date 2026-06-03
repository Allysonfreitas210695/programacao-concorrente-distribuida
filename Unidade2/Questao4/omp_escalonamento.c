#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Uso: %s <num_iteracoes> <num_threads>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int thread_count = atoi(argv[2]);
    int* thread_de_iter = (int*)malloc(n * sizeof(int));

    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < n; i++) {
        thread_de_iter[i] = omp_get_thread_num();
    }

    for (int t = 0; t < thread_count; t++) {
        int inicio = -1, fim = -1;
        printf("Thread %d: Iterações", t);
        int primeira = 1;
        for (int i = 0; i < n; i++) {
            if (thread_de_iter[i] == t) {
                if (inicio == -1) inicio = i;
                fim = i;
            } else if (inicio != -1 && fim != -1 && thread_de_iter[i] != t) {
                if (primeira) {
                    printf(" %d -- %d", inicio, fim);
                    primeira = 0;
                }
                inicio = -1; fim = -1;
            }
        }
        if (inicio != -1) printf(" %d -- %d", inicio, fim);
        printf("\n");
    }

    free(thread_de_iter);
    return 0;
}
