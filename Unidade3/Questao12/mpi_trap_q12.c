/* Questão 12 - Lista MPI
 * Cronometragem da regra trapezoidal com MPI_Reduce
 * para diferentes valores de n e p.
 *
 * Compile: mpicc -g -Wall -O2 -o mpi_trap_q12 mpi_trap_q12.c
 * Run:     mpiexec -n <p> ./mpi_trap_q12 <n>
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double Trap(double left, double right, int n, double h);
double f(double x);

int main(int argc, char* argv[]) {
    int my_rank, comm_sz, n, local_n;
    double a = 0.0, b = 1.0, h, local_a, local_b;
    double local_int, total_int;
    double start, finish, elapsed, max_elapsed;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc < 2) {
        if (my_rank == 0) fprintf(stderr, "Uso: %s <n>\n", argv[0]);
        MPI_Finalize(); return 1;
    }
    n = atoi(argv[1]);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h       = (b - a) / n;
    local_n = n / comm_sz;
    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    local_int = Trap(local_a, local_b, local_n, h);
    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    finish  = MPI_Wtime();
    elapsed = finish - start;

    /* Tempo máximo entre todos os processos = tempo de wall clock */
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("p=%d n=%d integral=%.15e tempo=%.6f s\n",
               comm_sz, n, total_int, max_elapsed);
    }

    MPI_Finalize();
    return 0;
}

double Trap(double left, double right, int n, double h) {
    double est = (f(left) + f(right)) / 2.0;
    for (int i = 1; i <= n - 1; i++) est += f(left + i * h);
    return est * h;
}

double f(double x) { return x * x; }
