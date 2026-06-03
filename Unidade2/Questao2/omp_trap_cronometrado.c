#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double f(double x) {
    return x * x;  /* f(x) = x^2 */
}

double Trap(double a, double b, int n, int thread_count) {
    double h = (b - a) / n;
    double local_int = 0.0;
    double inicio, fim;

    inicio = omp_get_wtime();

    #pragma omp parallel num_threads(thread_count) reduction(+:local_int)
    {
        int i;
        int my_rank = omp_get_thread_num();
        int local_n = n / thread_count;
        double local_a = a + my_rank * local_n * h;
        double local_b = local_a + local_n * h;
        double local_sum = (f(local_a) + f(local_b)) / 2.0;

        for (i = 1; i < local_n; i++) {
            double x = local_a + i * h;
            local_sum += f(x);
        }
        local_int += local_sum * h;

        #pragma omp barrier
    }

    fim = omp_get_wtime();
    printf("Tempo do bloco paralelo: %f segundos\n", fim - inicio);

    return local_int;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Uso: %s <n> <num_threads>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int thread_count = atoi(argv[2]);
    double a = 0.0, b = 1.0;

    double resultado = Trap(a, b, n, thread_count);
    printf("Resultado: %.6f\n", resultado);
    return 0;
}
