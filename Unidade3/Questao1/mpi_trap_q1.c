/* Questão 1 - Lista MPI
 * Modificação da regra trapezoidal para funcionar corretamente
 * mesmo quando comm_sz não divide n uniformemente.
 *
 * Compile: mpicc -g -Wall -o mpi_trap_q1 mpi_trap_q1.c
 * Run:     mpiexec -n <p> ./mpi_trap_q1
 */
#include <stdio.h>
#include <mpi.h>

double Trap(double left_endpt, double right_endpt, int trap_count, double base_len);
double f(double x);
void Get_input(int my_rank, double* a_p, double* b_p, int* n_p);

int main(void) {
    int my_rank, comm_sz, n, local_n;
    double a, b, h, local_a, local_b;
    double local_int, total_int;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    Get_input(my_rank, &a, &b, &n);

    h = (b - a) / n;

    /* MODIFICAÇÃO PRINCIPAL:
     * Distribui os trapézios restantes (n % comm_sz) entre os primeiros processos.
     * Processos com rank < remainder recebem (base_n + 1) trapézios.
     * Os demais recebem base_n trapézios.
     */
    int base_n   = n / comm_sz;
    int remainder = n % comm_sz;

    if (my_rank < remainder) {
        local_n = base_n + 1;
        local_a = a + my_rank * local_n * h;
    } else {
        local_n = base_n;
        local_a = a + (my_rank * base_n + remainder) * h;
    }
    local_b = local_a + local_n * h;

    local_int = Trap(local_a, local_b, local_n, h);

    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Com n = %d trapézios, estimativa da integral de %f a %f = %.15e\n",
               n, a, b, total_int);
    }

    MPI_Finalize();
    return 0;
}

void Get_input(int my_rank, double* a_p, double* b_p, int* n_p) {
    if (my_rank == 0) {
        printf("Entre com a, b e n:\n");
        scanf("%lf %lf %d", a_p, b_p, n_p);
    }
    MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_p, 1, MPI_INT,    0, MPI_COMM_WORLD);
}

double Trap(double left_endpt, double right_endpt, int trap_count, double base_len) {
    double estimate, x;
    int i;
    if (trap_count == 0) return 0.0;
    estimate = (f(left_endpt) + f(right_endpt)) / 2.0;
    for (i = 1; i <= trap_count - 1; i++) {
        x = left_endpt + i * base_len;
        estimate += f(x);
    }
    return estimate * base_len;
}

double f(double x) { return x * x; }
