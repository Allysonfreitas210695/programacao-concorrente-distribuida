/* Questão 6 - Lista MPI
 * (a) Allreduce via passagem em anel (ring)
 * (b) Somas de prefixos via anel
 *
 * Compile: mpicc -g -Wall -o mpi_ring_q6 mpi_ring_q6.c
 * Run:     mpiexec -n <p> ./mpi_ring_q6
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    srand(my_rank + 10);
    int my_val = rand() % 20;

    int dest   = (my_rank + 1) % comm_sz;
    int source = (my_rank - 1 + comm_sz) % comm_sz;
    int sendtag = 0, recvtag = 0;

    /* -------------------------------------------------------
     * (a) ALLREDUCE via anel
     * ------------------------------------------------------- */
    int sum = my_val;
    int temp_val = my_val;

    for (int i = 1; i < comm_sz; i++) {
        MPI_Sendrecv_replace(&temp_val, 1, MPI_INT,
                             dest,   sendtag,
                             source, recvtag,
                             MPI_COMM_WORLD, &status);
        sum += temp_val;
    }

    printf("[Anel Allreduce] Proc %d: my_val=%d, sum=%d\n", my_rank, my_val, sum);
    fflush(stdout);

    /* Verificação com MPI_Allreduce */
    int mpi_sum;
    MPI_Allreduce(&my_val, &mpi_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (my_rank == 0)
        printf("[MPI_Allreduce] sum=%d  (deve ser igual ao anel)\n\n", mpi_sum);

    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------------------------------------
     * (b) SOMAS DE PREFIXOS via anel
     * Cada processo q precisa da soma de todos os valores
     * dos processos com rank <= q.
     * Estratégia: processo q passa seu acumulador para q+1,
     * mas só acumula os valores de processos anteriores.
     * ------------------------------------------------------- */

    /* Reinicia temp_val com o valor original */
    temp_val = my_val;
    int prefix = my_val;

    /* O processo que enviou o valor original é rastreado pelo índice de volta */
    /* Cada iteração i: temp_val contém o valor do processo (my_rank - i) mod p */
    for (int i = 1; i < comm_sz; i++) {
        MPI_Sendrecv_replace(&temp_val, 1, MPI_INT,
                             dest,   sendtag,
                             source, recvtag,
                             MPI_COMM_WORLD, &status);
        /* O valor recebido veio do processo (my_rank - i) mod p */
        int sender_rank = (my_rank - i + comm_sz) % comm_sz;
        if (sender_rank < my_rank)
            prefix += temp_val;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("[Anel PrefixSum] Proc %d: my_val=%d, prefix_sum=%d\n", my_rank, my_val, prefix);
    fflush(stdout);

    /* Verificação com MPI_Scan */
    int mpi_prefix;
    MPI_Scan(&my_val, &mpi_prefix, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[MPI_Scan]       Proc %d: prefix_sum=%d\n", my_rank, mpi_prefix);

    MPI_Finalize();
    return 0;
}
