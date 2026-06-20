/* Questão 2 - Lista MPI
 * Modificação de mpi_output.c para imprimir na ordem dos ranks.
 * Processo 0 imprime primeiro, depois 1, 2, ...
 *
 * Compile: mpicc -g -Wall -o mpi_output_q2 mpi_output_q2.c
 * Run:     mpiexec -n <p> ./mpi_output_q2
 */
#include <stdio.h>
#include <mpi.h>

int main(void) {
    int my_rank, comm_sz;
    MPI_Status status;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Processo 0 imprime e envia um token para o próximo */
    if (my_rank == 0) {
        printf("Proc %d de %d > Does anyone have a toothpick?\n", my_rank, comm_sz);
        fflush(stdout);
        if (comm_sz > 1) {
            /* Envia token (qualquer valor) para o processo 1 */
            int token = 1;
            MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        }
    } else {
        /* Aguarda o token do processo anterior antes de imprimir */
        int token;
        MPI_Recv(&token, 1, MPI_INT, my_rank - 1, 0, MPI_COMM_WORLD, &status);
        printf("Proc %d de %d > Does anyone have a toothpick?\n", my_rank, comm_sz);
        fflush(stdout);
        /* Repassa o token para o próximo processo */
        if (my_rank < comm_sz - 1) {
            MPI_Send(&token, 1, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
