/*
 * Questão 4 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Programa MPI que:
 *   1. Processo 0 lê dois vetores (x, y) e um escalar do usuário.
 *   2. Distribui os vetores entre todos os processos com MPI_Scatter.
 *   3. Cada processo multiplica sua parte de x pelo escalar.
 *   4. Cada processo calcula a soma parcial dos quadrados de y (para a norma).
 *   5. Processo 0 coleta x*scalar com MPI_Gather e obtém a norma com MPI_Reduce.
 *
 * Restrição: n deve ser divisível por comm_sz (MPI_Scatter exige blocos iguais).
 *            A Questão 7 remove esta restrição com MPI_Scatterv.
 *
 * Compile: mpicc -g -Wall -o mpi_vec_q4 mpi_vec_q4.c -lm
 * Run:     mpiexec -n <p> ./mpi_vec_q4
 */

#include <stdio.h>    /* printf, scanf */
#include <stdlib.h>   /* malloc, free */
#include <math.h>     /* sqrt */
#include <mpi.h>      /* todas as funções MPI_ */

int main(void) {
    int my_rank, comm_sz;   /* rank do processo atual e total de processos */
    int n;                  /* tamanho global dos vetores */
    int local_n;            /* tamanho da fatia local (n / comm_sz) */
    double scalar;          /* escalar que multiplica o vetor x */

    /* Ponteiros para vetores GLOBAIS — alocados apenas no processo 0 */
    double *x = NULL, *y = NULL;

    /* Ponteiros para fatias LOCAIS de cada processo */
    double *local_x, *local_y;

    /* Onde processo 0 reunirá o vetor x*scalar após o Gather */
    double *result_x = NULL;

    /* Acumuladores para o cálculo da norma euclidiana de y */
    double local_norm_sq;   /* soma de quadrados na fatia local */
    double global_norm_sq;  /* soma global após Reduce */

    /* Inicializa o MPI */
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* ── FASE 1: Leitura dos dados (somente processo 0) ── */
    if (my_rank == 0) {
        printf("Digite n (ordem dos vetores, divisível por %d): ", comm_sz);
        scanf("%d", &n);
        printf("Digite o escalar: ");
        scanf("%lf", &scalar);

        /* Aloca os vetores globais apenas no processo 0 */
        x = malloc(n * sizeof(double));
        y = malloc(n * sizeof(double));

        printf("Digite os %d elementos do vetor x:\n", n);
        for (int i = 0; i < n; i++) scanf("%lf", &x[i]);

        printf("Digite os %d elementos do vetor y:\n", n);
        for (int i = 0; i < n; i++) scanf("%lf", &y[i]);
    }

    /* ── FASE 2: Distribuição de n e scalar ── */

    /*
     * MPI_Bcast: processo 0 envia 'n' para todos os outros.
     * Todos os processos precisam saber n para calcular local_n.
     */
    MPI_Bcast(&n,      1, MPI_INT,    0, MPI_COMM_WORLD);

    /*
     * MPI_Bcast: processo 0 envia 'scalar' para todos os outros.
     */
    MPI_Bcast(&scalar, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Cada processo calcula o tamanho de sua fatia */
    local_n = n / comm_sz;   /* divisão exata — n deve ser múltiplo de comm_sz */

    /* Aloca as fatias locais em cada processo */
    local_x = malloc(local_n * sizeof(double));
    local_y = malloc(local_n * sizeof(double));

    /* ── FASE 3: Scatter — distribui os vetores globais ── */

    /*
     * MPI_Scatter: divide x em blocos de 'local_n' elementos e envia
     * um bloco para cada processo (inclusive para o próprio processo 0).
     *
     * Parâmetros:
     *   x           → buffer de envio (só importa no processo 0)
     *   local_n     → quantos elementos enviar para cada processo
     *   MPI_DOUBLE  → tipo de dado
     *   local_x     → buffer de recepção de cada processo
     *   local_n     → quantos elementos cada processo recebe
     *   MPI_DOUBLE  → tipo de dado
     *   0           → processo raiz (quem tem os dados originais)
     *   MPI_COMM_WORLD
     *
     * Processo 0 recebe x[0..local_n-1]
     * Processo 1 recebe x[local_n..2*local_n-1]
     * ... e assim por diante.
     */
    MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, local_n, MPI_DOUBLE, local_y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ── FASE 4: Computação local ── */

    /* Operação 1: multiplica cada elemento local de x pelo escalar */
    for (int i = 0; i < local_n; i++)
        local_x[i] *= scalar;

    /* Operação 2: acumula soma dos quadrados de y para calcular a norma
     * ||y|| = sqrt(y[0]² + y[1]² + ... + y[n-1]²)
     * Cada processo calcula sua parcela da soma. */
    local_norm_sq = 0.0;
    for (int i = 0; i < local_n; i++)
        local_norm_sq += local_y[i] * local_y[i];

    /* ── FASE 5: Coleta dos resultados ── */

    /* Processo 0 precisa de espaço para reunir todos os blocos de local_x */
    if (my_rank == 0) result_x = malloc(n * sizeof(double));

    /*
     * MPI_Gather: inverso do MPI_Scatter — cada processo envia 'local_n'
     * elementos de local_x; processo 0 os concatena em result_x.
     *
     * Processo 0: result_x[0..local_n-1]           ← local_x do proc 0
     *             result_x[local_n..2*local_n-1]   ← local_x do proc 1
     *             ...
     */
    MPI_Gather(local_x, local_n, MPI_DOUBLE, result_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*
     * MPI_Reduce: soma todas as somas parciais dos quadrados.
     * Resultado: global_norm_sq = local_norm_sq(0) + local_norm_sq(1) + ...
     * Armazenado somente no processo 0.
     */
    MPI_Reduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ── FASE 6: Impressão do resultado (somente processo 0) ── */
    if (my_rank == 0) {
        printf("\nResultado de x * %.2f:\n", scalar);
        for (int i = 0; i < n; i++) printf("%.4f ", result_x[i]);
        printf("\n");

        /* sqrt da soma dos quadrados = norma euclidiana */
        printf("Norma de y: %.6f\n", sqrt(global_norm_sq));

        /* Libera memória dos vetores globais (alocados só no proc 0) */
        free(x); free(y); free(result_x);
    }

    /* Libera memória das fatias locais (todos os processos) */
    free(local_x);
    free(local_y);

    MPI_Finalize();
    return 0;
}
