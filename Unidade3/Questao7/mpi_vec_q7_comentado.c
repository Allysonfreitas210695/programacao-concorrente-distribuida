/*
 * Questão 7 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Modificar a Questão 4 para suportar n NÃO divisível por comm_sz.
 * Substitui MPI_Scatter/MPI_Gather por MPI_Scatterv/MPI_Gatherv, que permitem
 * blocos de tamanhos DIFERENTES para cada processo.
 *
 * Diferença da Q4:
 *   Q4: todos os processos recebem exatamente (n/comm_sz) elementos.
 *   Q7: processos 0..(rem-1) recebem (base+1) e demais recebem (base) elementos.
 *       Isso funciona para qualquer n.
 *
 * Compile: mpicc -g -Wall -o mpi_vec_q7 mpi_vec_q7.c -lm
 * Run:     mpiexec -n <p> ./mpi_vec_q7
 */

#include <stdio.h>    /* printf, scanf */
#include <stdlib.h>   /* malloc, free */
#include <math.h>     /* sqrt */
#include <mpi.h>

int main(void) {
    int my_rank, comm_sz, n;
    double scalar;

    /* Vetores globais: alocados apenas no processo 0 */
    double *x = NULL, *y = NULL;

    /* Fatias locais de cada processo */
    double *local_x, *local_y;

    /* Onde processo 0 reunirá x*scalar após Gatherv */
    double *result_x = NULL;

    double local_norm_sq, global_norm_sq;

    /* Arrays exigidos pelo MPI_Scatterv e MPI_Gatherv:
     *   sendcounts[q] → quantos elementos o processo q recebe/envia
     *   displs[q]     → deslocamento (offset) no vetor global para o processo q */
    int *sendcounts = NULL, *displs = NULL;

    int local_n;   /* número de elementos neste processo */

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* ── FASE 1: Leitura (somente processo 0) ── */
    if (my_rank == 0) {
        /* n pode ser QUALQUER valor, não precisa ser divisível por comm_sz */
        printf("Digite n (qualquer valor): ");
        scanf("%d", &n);
        printf("Digite o escalar: ");
        scanf("%lf", &scalar);

        x = malloc(n * sizeof(double));
        y = malloc(n * sizeof(double));

        printf("Digite os %d elementos do vetor x:\n", n);
        for (int i = 0; i < n; i++) scanf("%lf", &x[i]);

        printf("Digite os %d elementos do vetor y:\n", n);
        for (int i = 0; i < n; i++) scanf("%lf", &y[i]);
    }

    /* Distribui n e scalar para todos */
    MPI_Bcast(&n,      1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&scalar, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ── FASE 2: Calcular distribuição desigual ── */

    /* Aloca os arrays de controle em todos os processos */
    sendcounts = malloc(comm_sz * sizeof(int));
    displs     = malloc(comm_sz * sizeof(int));

    int base = n / comm_sz;   /* fatia mínima garantida a cada processo */
    int rem  = n % comm_sz;   /* elementos "extras" a distribuir */

    int offset = 0;           /* ponteiro de deslocamento acumulado */
    for (int q = 0; q < comm_sz; q++) {
        /* Processos 0, 1, ..., rem-1 recebem um elemento extra */
        sendcounts[q] = (q < rem) ? base + 1 : base;

        /* Deslocamento no vetor global onde começa a fatia do processo q */
        displs[q] = offset;

        /* Avança o ponteiro pelo tamanho da fatia do processo q */
        offset += sendcounts[q];
    }

    /* Este processo recebe sendcounts[my_rank] elementos */
    local_n = sendcounts[my_rank];

    /* Aloca as fatias locais com o tamanho correto */
    local_x = malloc(local_n * sizeof(double));
    local_y = malloc(local_n * sizeof(double));

    /* ── FASE 3: Scatter com blocos desiguais ── */

    /*
     * MPI_Scatterv: versão variante do MPI_Scatter.
     *   Processo q recebe sendcounts[q] elementos a partir de x[displs[q]].
     *
     * Parâmetros extras em relação ao MPI_Scatter:
     *   sendcounts → array com a quantidade para cada processo
     *   displs     → array com o deslocamento no buffer de envio para cada processo
     *
     * Equivalente a:
     *   proc 0 recebe x[displs[0] .. displs[0]+sendcounts[0]-1]
     *   proc 1 recebe x[displs[1] .. displs[1]+sendcounts[1]-1]
     *   ...
     */
    MPI_Scatterv(x, sendcounts, displs, MPI_DOUBLE,
                 local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(y, sendcounts, displs, MPI_DOUBLE,
                 local_y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ── FASE 4: Computação local ── */

    /* Multiplica cada elemento local de x pelo escalar */
    for (int i = 0; i < local_n; i++) local_x[i] *= scalar;

    /* Calcula soma parcial dos quadrados de y (para a norma) */
    local_norm_sq = 0.0;
    for (int i = 0; i < local_n; i++) local_norm_sq += local_y[i] * local_y[i];

    /* ── FASE 5: Gather com blocos desiguais ── */

    if (my_rank == 0) result_x = malloc(n * sizeof(double));

    /*
     * MPI_Gatherv: inverso do MPI_Scatterv.
     *   Cada processo q envia local_n elementos de local_x.
     *   Processo 0 os coloca em result_x a partir de displs[q].
     *
     * Reconstrói o vetor global na mesma ordem original.
     */
    MPI_Gatherv(local_x, local_n, MPI_DOUBLE,
                result_x, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Reduz as somas de quadrados para obter a norma global */
    MPI_Reduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ── FASE 6: Impressão dos resultados (somente processo 0) ── */
    if (my_rank == 0) {
        printf("\nResultado de x * %.2f:\n", scalar);
        for (int i = 0; i < n; i++) printf("%.4f ", result_x[i]);
        printf("\nNorma de y: %.6f\n", sqrt(global_norm_sq));
        free(x); free(y); free(result_x);
    }

    free(local_x); free(local_y);
    free(sendcounts); free(displs);
    MPI_Finalize();
    return 0;
}

/*
 * EXEMPLO de distribuição com n=7 e comm_sz=3:
 *   base=2, rem=1
 *   Proc 0: sendcounts[0]=3, displs[0]=0  → x[0,1,2]
 *   Proc 1: sendcounts[1]=2, displs[1]=3  → x[3,4]
 *   Proc 2: sendcounts[2]=2, displs[2]=5  → x[5,6]
 *   Total: 3+2+2 = 7 ✓
 */
