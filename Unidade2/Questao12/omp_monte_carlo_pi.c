#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Uso: %s <num_lancamentos> <num_threads>\n", argv[0]);
        return 1;
    }

    long long num_lancamentos = atoll(argv[1]);
    int thread_count          = atoi(argv[2]);
    long long qtd_no_circulo  = 0;

    double inicio = omp_get_wtime();

    #pragma omp parallel num_threads(thread_count) \
            reduction(+:qtd_no_circulo)
    {
        int tid = omp_get_thread_num();

        /*
         * Semente diferente por thread para sequências independentes.
         * unsigned int na pilha é thread-local, portanto seguro.
         */
        unsigned int semente = (unsigned int)(12345 + tid * 9999);

        long long lancamentos_locais = num_lancamentos / thread_count;

        if (tid == 0)
            lancamentos_locais += num_lancamentos % thread_count;

        long long local_circulo = 0;

        for (long long i = 0; i < lancamentos_locais; i++) {
            double x = (double)rand_r(&semente) / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand_r(&semente) / RAND_MAX * 2.0 - 1.0;
            if (x * x + y * y <= 1.0)
                local_circulo++;
        }

        qtd_no_circulo += local_circulo;
    }

    double fim = omp_get_wtime();

    double estimativa_pi = 4.0 * qtd_no_circulo / (double)num_lancamentos;

    printf("Lancamentos  : %lld\n", num_lancamentos);
    printf("No circulo   : %lld\n", qtd_no_circulo);
    printf("Estimativa pi: %.8f\n", estimativa_pi);
    printf("pi real      : %.8f\n", M_PI);
    printf("Erro         : %.8f\n", fabs(estimativa_pi - M_PI));
    printf("Threads      : %d\n", thread_count);
    printf("Tempo        : %.4f segundos\n", fim - inicio);

    return 0;
}
