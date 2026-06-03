/* Inclui funções de entrada/saída padrão (printf) */
#include <stdio.h>
/* Inclui funções de alocação de memória (malloc, calloc, free) e atoi */
#include <stdlib.h>
/* Inclui a API OpenMP (pragma omp, omp_get_wtime) */
#include <omp.h>

int main(int argc, char* argv[]) {
    /* Verifica se foram passados os 3 argumentos obrigatórios */
    if (argc < 4) {
        printf("Uso: %s <num_elementos> <num_buckets> <num_threads>\n", argv[0]);
        return 1; /* Encerra com erro se faltar argumento */
    }

    /* n: quantidade total de elementos a serem contabilizados no histograma */
    int n            = atoi(argv[1]);

    /* num_buckets: número de intervalos (categorias) do histograma */
    int num_buckets  = atoi(argv[2]);

    /* thread_count: número de threads OpenMP que vão trabalhar em paralelo */
    int thread_count = atoi(argv[3]);

    /* Aloca o array de dados de entrada com n inteiros */
    int* dados = (int*)malloc(n * sizeof(int));

    /* Inicializa o gerador de números aleatórios com semente fixa (42)
     * para que os resultados sejam reproduzíveis */
    srand(42);

    /* Preenche o array com valores aleatórios no intervalo [0, num_buckets-1] */
    for (int i = 0; i < n; i++)
        dados[i] = rand() % num_buckets; /* Cada elemento cai em algum bucket */

    /* Aloca o histograma global com num_buckets posições, todas zeradas.
     * calloc já inicializa a memória com zero (diferente de malloc) */
    int* hist = (int*)calloc(num_buckets, sizeof(int));

    /* Marca o instante de início para medir o tempo da etapa paralela */
    double inicio = omp_get_wtime();

    /* ----------------------------------------------------------------
     * REGIÃO PARALELA
     * Estratégia: cada thread usa seu próprio histograma local para
     * evitar condição de corrida. Ao final, soma ao histograma global
     * uma única vez por thread (usando critical), reduzindo a contenção.
     * ---------------------------------------------------------------- */
    #pragma omp parallel num_threads(thread_count)
    {
        /* Cada thread aloca e zera seu próprio histograma local.
         * Isso evita que threads disputem as mesmas posições do hist global */
        int* hist_local = (int*)calloc(num_buckets, sizeof(int));

        /* Divide as iterações do laço entre as threads automaticamente.
         * Cada thread processa uma fatia de dados[0..n-1] */
        #pragma omp for
        for (int i = 0; i < n; i++) {
            /* Incrementa o bucket correspondente ao valor dados[i]
             * no histograma LOCAL desta thread (sem risco de corrida) */
            hist_local[dados[i]]++;
        }
        /* Barreira implícita aqui: todas as threads terminam o for antes de continuar */

        /* SEÇÃO CRÍTICA: apenas uma thread por vez soma seu histograma
         * local ao histograma global. Isso evita race condition em hist[].
         * Como é feito uma única vez por thread (e não por elemento),
         * o overhead é muito menor do que usar critical dentro do for */
        #pragma omp critical
        {
            for (int b = 0; b < num_buckets; b++)
                hist[b] += hist_local[b]; /* Acumula bucket a bucket no global */
        }

        /* Libera a memória do histograma local desta thread */
        free(hist_local);
    } /* Fim da região paralela — barreira implícita aguarda todas as threads */

    /* Marca o instante de fim da etapa paralela */
    double fim = omp_get_wtime();

    /* Imprime o cabeçalho com os parâmetros usados */
    printf("Histograma (%d elementos, %d buckets, %d threads):\n",
           n, num_buckets, thread_count);

    /* Imprime cada bucket e sua contagem */
    for (int b = 0; b < num_buckets; b++) {
        printf("  Bucket [%3d]: %d\n", b, hist[b]);
    }

    /* Imprime o tempo total da etapa paralela */
    printf("Tempo: %.4f segundos\n", fim - inicio);

    /* Libera o array de dados de entrada */
    free(dados);

    /* Libera o histograma global */
    free(hist);

    return 0; /* Programa encerrado com sucesso */
}
