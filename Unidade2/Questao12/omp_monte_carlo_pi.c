#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Uso: %s <num_lancamentos> <num_threads>\n", argv[0]);
        return 1; /* Encerra com erro se faltar argumento */
    }

    /* Lê o total de lançamentos como long long (pode ser bilhões — não cabe em int) */
    long long num_lancamentos = atoll(argv[1]);

    /* Lê o número de threads a serem usadas */
    int thread_count = atoi(argv[2]);

    /* Contador global de quantos dardos caíram dentro do círculo de raio 1.
     * Será somado via reduction — começa em zero */
    long long qtd_no_circulo = 0;

    /* Marca o início da cronometragem */
    double inicio = omp_get_wtime();

    /* ----------------------------------------------------------------
     * REGIÃO PARALELA com REDUCTION
     *
     * Cada thread vai:
     *   1. Gerar sua própria sequência de números aleatórios (semente privada)
     *   2. Lançar sua fatia de dardos
     *   3. Contar quantos caíram no círculo (local_circulo)
     * Ao final, o reduction soma todos os local_circulo em qtd_no_circulo.
     * ---------------------------------------------------------------- */
    #pragma omp parallel num_threads(thread_count) \
            reduction(+:qtd_no_circulo)
    {
        /* Obtém o identificador único desta thread (0, 1, 2, ...) */
        int tid = omp_get_thread_num();

        /* Cada thread usa uma semente diferente para gerar sequências
         * aleatórias independentes entre si.
         * unsigned int fica na pilha (stack), portanto é automaticamente
         * privado para cada thread — não há risco de race condition */
        unsigned int semente = (unsigned int)(12345 + tid * 9999);

        /* Divide o total de lançamentos igualmente entre as threads */
        long long lancamentos_locais = num_lancamentos / thread_count;

        /* A thread 0 absorve o resto da divisão inteira para garantir
         * que a soma total de lançamentos seja exatamente num_lancamentos */
        if (tid == 0)
            lancamentos_locais += num_lancamentos % thread_count;

        /* Contador local (privado por thread) de acertos no círculo */
        long long local_circulo = 0;

        /* Laço de Monte Carlo: lança lancamentos_locais dardos */
        for (long long i = 0; i < lancamentos_locais; i++) {
            /* Gera coordenada x aleatória no intervalo [-1.0, +1.0].
             * rand_r é thread-safe: usa &semente como estado local */
            double x = (double)rand_r(&semente) / RAND_MAX * 2.0 - 1.0;

            /* Gera coordenada y aleatória no intervalo [-1.0, +1.0] */
            double y = (double)rand_r(&semente) / RAND_MAX * 2.0 - 1.0;

            /* Verifica se o ponto (x, y) está dentro do círculo de raio 1.
             * Condição: x² + y² ≤ 1  (evita sqrt para maior desempenho) */
            if (x * x + y * y <= 1.0)
                local_circulo++; /* Dardo caiu dentro do círculo */
        }

        /* Acumula o resultado local no contador global.
         * O reduction garante que isso seja feito de forma segura */
        qtd_no_circulo += local_circulo;
    } /* Fim da região paralela — reduction soma todos os local_circulo */

    /* Marca o fim da cronometragem */
    double fim = omp_get_wtime();

    /* Calcula a estimativa de π usando a fórmula:
     * π ≈ 4 × (pontos_no_círculo / total_de_pontos)
     * Isso vem da razão área_círculo / área_quadrado = π/4 */
    double estimativa_pi = 4.0 * qtd_no_circulo / (double)num_lancamentos;

    /* Exibe o total de lançamentos realizados */
    printf("Lancamentos  : %lld\n", num_lancamentos);

    /* Exibe quantos dardos caíram dentro do círculo */
    printf("No circulo   : %lld\n", qtd_no_circulo);

    /* Exibe a estimativa calculada de π com 8 casas decimais */
    printf("Estimativa pi: %.8f\n", estimativa_pi);

    /* Exibe o valor real de π (constante M_PI da biblioteca math.h) */
    printf("pi real      : %.8f\n", M_PI);

    /* Exibe a diferença absoluta entre a estimativa e o valor real */
    printf("Erro         : %.8f\n", fabs(estimativa_pi - M_PI));

    /* Exibe quantas threads foram usadas */
    printf("Threads      : %d\n", thread_count);

    /* Exibe o tempo total de execução da região paralela */
    printf("Tempo        : %.4f segundos\n", fim - inicio);

    return 0; /* Programa encerrado com sucesso */
}
