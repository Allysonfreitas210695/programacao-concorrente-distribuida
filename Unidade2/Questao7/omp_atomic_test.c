#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Uso: %s <n> <num_threads>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int thread_count = atoi(argv[2]);
    double tempo_inicio, tempo_fim;

    tempo_inicio = omp_get_wtime();

    #pragma omp parallel num_threads(1)
    {
        int i;
        double minha_soma = 0.0;
        for (i = 0; i < n; i++) {
            #pragma omp atomic
            minha_soma += sin(i);
        }
    } 

    tempo_fim = omp_get_wtime();
    printf("1 thread: %.4f segundos\n", tempo_fim - tempo_inicio);

    tempo_inicio = omp_get_wtime();

    #pragma omp parallel num_threads(thread_count)
    {
        int i;
        double minha_soma = 0.0;

        for (i = 0; i < n; i++) {
            /* atomic protege minha_soma, mas como é privada de cada thread,
             * não há disputa entre threads — executam em paralelo */
            #pragma omp atomic
            minha_soma += sin(i);
        }
    }

    tempo_fim = omp_get_wtime();
    printf("%d threads: %.4f segundos\n", thread_count, tempo_fim - tempo_inicio);

    return 0;
}
