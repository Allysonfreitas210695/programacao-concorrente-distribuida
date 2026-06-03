/* Inclui funções de entrada/saída padrão (printf) */
#include <stdio.h>
/* Inclui funções de alocação de memória e conversão (atoi) */
#include <stdlib.h>
/* Inclui funções matemáticas (sin) */
#include <math.h>
/* Inclui a API OpenMP (pragma omp, omp_get_wtime, etc.) */
#include <omp.h>

int main(int argc, char* argv[]) {
    /* Verifica se o usuário passou os dois argumentos obrigatórios */
    if (argc < 3) {
        printf("Uso: %s <n> <num_threads>\n", argv[0]);
        return 1; /* Encerra o programa com código de erro */
    }

    /* Converte o 1º argumento (quantidade de iterações) para inteiro */
    int n = atoi(argv[1]);

    /* Converte o 2º argumento (número de threads) para inteiro */
    int thread_count = atoi(argv[2]);

    /* Variáveis para armazenar o início e o fim do cronômetro */
    double tempo_inicio, tempo_fim;

    /* ----------------------------------------------------------------
     * TESTE 1: execução com apenas 1 thread
     * Objetivo: medir o tempo de referência (baseline serial)
     * ---------------------------------------------------------------- */

    /* Marca o instante de início do teste com 1 thread */
    tempo_inicio = omp_get_wtime();

    /* Cria uma região paralela forçada a usar somente 1 thread */
    #pragma omp parallel num_threads(1)
    {
        int i;

        /* minha_soma declarada DENTRO do bloco paralelo:
         * cada thread tem sua própria cópia privada (variável local) */
        double minha_soma = 0.0;

        /* Laço que acumula seno de cada índice em minha_soma */
        for (i = 0; i < n; i++) {
            /* #pragma omp atomic garante que a operação += seja
             * atômica (indivisível), protegendo minha_soma de
             * condições de corrida. Como minha_soma é privada aqui,
             * o atomic não causa contenção — cada thread protege
             * sua própria variável separada */
            #pragma omp atomic
            minha_soma += sin(i); /* Acumula sin(i) de forma atômica */
        }
    } /* Todas as threads chegam aqui antes de continuar (barreira implícita) */

    /* Marca o instante de fim do teste com 1 thread */
    tempo_fim = omp_get_wtime();

    /* Imprime o tempo gasto com 1 thread */
    printf("1 thread: %.4f segundos\n", tempo_fim - tempo_inicio);

    /* ----------------------------------------------------------------
     * TESTE 2: execução com thread_count threads
     * Objetivo: verificar se atomic em variáveis DIFERENTES causa
     * serialização (como critical) ou permite execução paralela.
     * Se os tempos de 1 thread e N threads forem similares → cada
     * atomic protege apenas a sua variável (seções críticas independentes).
     * Se N threads for muito mais lento → atomic seria uma seção crítica
     * global, causando serialização.
     * ---------------------------------------------------------------- */

    /* Marca o instante de início do teste com múltiplas threads */
    tempo_inicio = omp_get_wtime();

    /* Cria uma região paralela com thread_count threads */
    #pragma omp parallel num_threads(thread_count)
    {
        int i;

        /* Cada thread tem a sua própria cópia de minha_soma (local na pilha) */
        double minha_soma = 0.0;

        /* Mesmo laço: cada thread executa n iterações na sua própria cópia */
        for (i = 0; i < n; i++) {
            /* atomic protege minha_soma, mas como é privada de cada thread,
             * não há disputa entre threads — executam em paralelo */
            #pragma omp atomic
            minha_soma += sin(i);
        }
    } /* Barreira implícita: aguarda todas as threads terminarem */

    /* Marca o instante de fim do teste com múltiplas threads */
    tempo_fim = omp_get_wtime();

    /* Imprime o tempo gasto com thread_count threads.
     * Resultado esperado: tempo ≈ ao de 1 thread (ou menor),
     * confirmando que atomic em variáveis diferentes NÃO serializa */
    printf("%d threads: %.4f segundos\n", thread_count, tempo_fim - tempo_inicio);

    return 0; /* Programa encerrado com sucesso */
}
