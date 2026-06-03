#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double f(double x) { return x * x; }

double Trap(double a, double b, int n, int thread_count, int* iteracoes) {
    double h = (b - a) / n;
    double sum = (f(a) + f(b)) / 2.0;

    #pragma omp parallel for num_threads(thread_count) \
            schedule(runtime) reduction(+:sum)
    for (int i = 1; i < n; i++) {
        iteracoes[i] = omp_get_thread_num();
        double x = a + i * h;
        sum += f(x);
    }
    return sum * h;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Uso: %s <n> <num_threads>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int thread_count = atoi(argv[2]);
    double a = 0.0, b = 1.0;

    int* iteracoes = (int*)malloc(n * sizeof(int));

    double resultado = Trap(a, b, n, thread_count, iteracoes);
    printf("Resultado: %.6f\n", resultado);

    for (int t = 0; t < thread_count; t++) {
        printf("Thread %d: ", t);
        for (int i = 1; i < n; i++)
            if (iteracoes[i] == t)
                printf("%d ", i);
        printf("\n");
    }

    free(iteracoes);
    return 0;
}
