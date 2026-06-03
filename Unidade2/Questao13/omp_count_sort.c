#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void Count_sort_paralelo(int a[], int n, int thread_count) {
    int* temp = (int*)malloc(n * sizeof(int));

    #pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (a[j] < a[i])
                count++;
            else if (a[j] == a[i] && j < i)
                count++;
        }
        temp[count] = a[i];
    }

    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < n; i++)
        a[i] = temp[i];

    free(temp);
}

void Count_sort_serial(int a[], int n) {
    int* temp = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (a[j] < a[i]) count++;
            else if (a[j] == a[i] && j < i) count++;
        }
        temp[count] = a[i];
    }
    memcpy(a, temp, n * sizeof(int));
    free(temp);
}

int cmp(const void* x, const void* y) {
    return (*(int*)x - *(int*)y);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Uso: %s <n> <num_threads>\n", argv[0]);
        return 1;
    }

    int n            = atoi(argv[1]);
    int thread_count = atoi(argv[2]);

    int* a1 = (int*)malloc(n * sizeof(int));
    int* a2 = (int*)malloc(n * sizeof(int));
    int* a3 = (int*)malloc(n * sizeof(int));

    srand(42);
    for (int i = 0; i < n; i++)
        a1[i] = a2[i] = a3[i] = rand() % (n * 2);

    double t_inicio, t_fim;

    t_inicio = omp_get_wtime();
    Count_sort_serial(a1, n);
    t_fim = omp_get_wtime();
    printf("Count sort serial  : %.4f segundos\n", t_fim - t_inicio);

    t_inicio = omp_get_wtime();
    Count_sort_paralelo(a2, n, thread_count);
    t_fim = omp_get_wtime();
    printf("Count sort paralelo: %.4f segundos (%d threads)\n",
           t_fim - t_inicio, thread_count);

    t_inicio = omp_get_wtime();
    qsort(a3, n, sizeof(int), cmp);
    t_fim = omp_get_wtime();
    printf("qsort (serial)     : %.4f segundos\n", t_fim - t_inicio);

    int correto = 1;
    for (int i = 0; i < n; i++)
        if (a2[i] != a3[i]) { correto = 0; break; }
    printf("Resultado paralelo %s\n", correto ? "CORRETO" : "INCORRETO");

    free(a1); free(a2); free(a3);
    return 0;
}
