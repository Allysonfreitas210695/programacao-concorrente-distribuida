/* Questão 11 - Lista MPI
 * Função Get_input para a regra trapezoidal usando
 * MPI_Pack no processo 0 e MPI_Unpack nos demais.
 *
 * Compile: mpicc -g -Wall -o mpi_trap_q11 mpi_trap_q11.c
 * Run:     mpiexec -n <p> ./mpi_trap_q11
 */
#include <stdio.h>
#include <mpi.h>

double Trap(double left, double right, int n, double h);
double f(double x);
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p);

int main(void) {
    int my_rank, comm_sz, n, local_n;
    double a, b, h, local_a, local_b;
    double local_int, total_int;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    Get_input(my_rank, comm_sz, &a, &b, &n);

    h       = (b - a) / n;
    local_n = n / comm_sz;
    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;

    local_int = Trap(local_a, local_b, local_n, h);
    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
        printf("Integral de %f a %f com n=%d: %.15e\n", a, b, n, total_int);

    MPI_Finalize();
    return 0;
}

/* MODIFICAÇÃO PRINCIPAL: Get_input com MPI_Pack/MPI_Unpack */
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p) {
    /* Buffer suficientemente grande para 2 doubles + 1 int */
    char pack_buf[100];
    int position = 0;

    if (my_rank == 0) {
        printf("Entre com a, b e n:\n");
        scanf("%lf %lf %d", a_p, b_p, n_p);

        /* Empacota os três valores no buffer */
        MPI_Pack(a_p, 1, MPI_DOUBLE, pack_buf, 100, &position, MPI_COMM_WORLD);
        MPI_Pack(b_p, 1, MPI_DOUBLE, pack_buf, 100, &position, MPI_COMM_WORLD);
        MPI_Pack(n_p, 1, MPI_INT,    pack_buf, 100, &position, MPI_COMM_WORLD);
    }

    /* Broadcast do buffer empacotado com tipo MPI_PACKED */
    MPI_Bcast(pack_buf, 100, MPI_PACKED, 0, MPI_COMM_WORLD);

    if (my_rank != 0) {
        /* Desempacota na mesma ordem em que foi empacotado */
        position = 0;
        MPI_Unpack(pack_buf, 100, &position, a_p, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Unpack(pack_buf, 100, &position, b_p, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Unpack(pack_buf, 100, &position, n_p, 1, MPI_INT,    MPI_COMM_WORLD);
    }
}

double Trap(double left, double right, int n, double h) {
    double est = (f(left) + f(right)) / 2.0;
    for (int i = 1; i <= n - 1; i++) est += f(left + i * h);
    return est * h;
}

double f(double x) { return x * x; }
