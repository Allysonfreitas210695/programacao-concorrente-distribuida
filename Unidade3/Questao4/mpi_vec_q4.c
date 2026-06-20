/* Questão 4 - Lista MPI
 * Programa que:
 *   - Lê dois vetores e um escalar no processo 0
 *   - Distribui entre os processos (n divisível por comm_sz)
 *   - Multiplica o primeiro vetor pelo escalar
 *   - Calcula a norma do segundo vetor
 *   - Coleta e imprime resultados no processo 0
 *
 * Compile: mpicc -g -Wall -o mpi_vec_q4 mpi_vec_q4.c -lm
 * Run:     mpiexec -n <p> ./mpi_vec_q4
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(void) {
    int my_rank, comm_sz, n, local_n;
    double scalar;
    double *x = NULL, *y = NULL;           /* vetores globais (apenas proc 0) */
    double *local_x, *local_y;            /* vetores locais */
    double *result_x = NULL;              /* resultado global de x*scalar */
    double local_norm_sq, global_norm_sq; /* para norma de y */

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Processo 0 lê os dados */
    if (my_rank == 0) {
        printf("Digite n (ordem dos vetores, divisível por %d): ", comm_sz);
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

    /* Broadcast de n e scalar */
    MPI_Bcast(&n,      1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&scalar, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_n = n / comm_sz;
    local_x = malloc(local_n * sizeof(double));
    local_y = malloc(local_n * sizeof(double));

    /* Distribui os vetores */
    MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, local_n, MPI_DOUBLE, local_y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Operação 1: multiplica local_x pelo escalar */
    for (int i = 0; i < local_n; i++)
        local_x[i] *= scalar;

    /* Operação 2: soma parcial dos quadrados para a norma de y */
    local_norm_sq = 0.0;
    for (int i = 0; i < local_n; i++)
        local_norm_sq += local_y[i] * local_y[i];

    /* Coleta vetor x*scalar no processo 0 */
    if (my_rank == 0) result_x = malloc(n * sizeof(double));
    MPI_Gather(local_x, local_n, MPI_DOUBLE, result_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Reduz a soma dos quadrados para a norma */
    MPI_Reduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Processo 0 imprime resultados */
    if (my_rank == 0) {
        printf("\nResultado de x * %.2f:\n", scalar);
        for (int i = 0; i < n; i++) printf("%.4f ", result_x[i]);
        printf("\n");
        printf("Norma de y: %.6f\n", sqrt(global_norm_sq));

        free(x); free(y); free(result_x);
    }

    free(local_x);
    free(local_y);
    MPI_Finalize();
    return 0;
}
