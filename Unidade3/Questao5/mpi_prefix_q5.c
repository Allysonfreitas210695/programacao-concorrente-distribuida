/* Questão 5 - Lista MPI
 * (a) Algoritmo serial de somas de prefixos
 * (b) Algoritmo paralelo sem MPI_Scan (k fases, n = 2^k processos)
 * (c) Versão com MPI_Scan
 *
 * Compile: mpicc -g -Wall -o mpi_prefix_q5 mpi_prefix_q5.c
 * Run:     mpiexec -n <p> ./mpi_prefix_q5
 *
 * Nota: para (b), p deve ser potência de 2.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* (a) Algoritmo SERIAL de somas de prefixos — O(n) */
void serial_prefix_sum(double *x, double *prefix, int n) {
    prefix[0] = x[0];
    for (int i = 1; i < n; i++)
        prefix[i] = prefix[i - 1] + x[i];
}

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Cada processo gera um valor aleatório */
    srand(my_rank + 42);
    double my_val = (double)(rand() % 100);

    /* -------------------------------------------------------
     * (b) Paralelo SEM MPI_Scan: algoritmo de varredura em log2(p) fases.
     *
     * Cada fase 'half': processo q envia seu acumulador para q+half
     *                   e recebe de q-half (se existir).
     * Após log2(p) fases, processo q contém x[0]+...+x[q].
     *
     * Invariante: ao final da fase com 'half', cur[q] acumula a soma
     * dos half processos anteriores mais o próprio. Após a última fase
     * (half = p/2): cur[q] = x[0]+...+x[q]. ✓
     * ------------------------------------------------------- */
    double cur = my_val;
    for (int half = 1; half < comm_sz; half <<= 1) {
        double received = 0.0;
        /* MPI_PROC_NULL descarta send/recv silenciosamente — evita deadlock */
        int dest   = (my_rank + half < comm_sz) ? my_rank + half : MPI_PROC_NULL;
        int source = (my_rank - half >= 0)      ? my_rank - half : MPI_PROC_NULL;
        MPI_Sendrecv(&cur,      1, MPI_DOUBLE, dest,   0,
                     &received, 1, MPI_DOUBLE, source, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (source != MPI_PROC_NULL)
            cur += received;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double prefix_val = cur;

    /* Impressão ordenada por rank (token ring) */
    if (my_rank == 0) {
        printf("=== (b) Paralelo sem MPI_Scan ===\n");
        printf("Proc %d: valor=%.0f, prefix_sum=%.0f\n", my_rank, my_val, prefix_val);
        fflush(stdout);
        if (comm_sz > 1) { int t = 1; MPI_Send(&t, 1, MPI_INT, 1, 99, MPI_COMM_WORLD); }
    } else {
        int t;
        MPI_Recv(&t, 1, MPI_INT, my_rank - 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Proc %d: valor=%.0f, prefix_sum=%.0f\n", my_rank, my_val, prefix_val);
        fflush(stdout);
        if (my_rank < comm_sz - 1)
            MPI_Send(&t, 1, MPI_INT, my_rank + 1, 99, MPI_COMM_WORLD);
    }

    /* -------------------------------------------------------
     * (c) Com MPI_Scan
     * ------------------------------------------------------- */
    double scan_result;
    MPI_Scan(&my_val, &scan_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) {
        printf("\n=== (c) Com MPI_Scan ===\n");
        printf("Proc %d: valor=%.0f, MPI_Scan=%.0f\n", my_rank, my_val, scan_result);
        fflush(stdout);
        if (comm_sz > 1) { int t = 1; MPI_Send(&t, 1, MPI_INT, 1, 98, MPI_COMM_WORLD); }
    } else {
        int t;
        MPI_Recv(&t, 1, MPI_INT, my_rank - 1, 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Proc %d: valor=%.0f, MPI_Scan=%.0f\n", my_rank, my_val, scan_result);
        fflush(stdout);
        if (my_rank < comm_sz - 1)
            MPI_Send(&t, 1, MPI_INT, my_rank + 1, 98, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
