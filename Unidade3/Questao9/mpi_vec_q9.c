/* Questão 9 - Lista MPI
 * Modificação de Read_vector e Print_vector para usar
 * MPI_Type_contiguous com count=1 em MPI_Scatter e MPI_Gather.
 *
 * Trecho principal: apenas as funções modificadas de mpi_vector_add.c
 *
 * Compile: mpicc -g -Wall -o mpi_vec_q9 mpi_vec_q9.c
 * Run:     mpiexec -n <p> ./mpi_vec_q9
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_vectors(double** lx, double** ly, double** lz, int local_n, MPI_Comm comm);
void Read_vector(double local_a[], int local_n, int n, char vec_name[], int my_rank, MPI_Comm comm);
void Print_vector(double local_b[], int local_n, int n, char title[], int my_rank, MPI_Comm comm);
void Parallel_vector_sum(double lx[], double ly[], double lz[], int local_n);

int main(void) {
    int n, local_n, comm_sz, my_rank;
    double *local_x, *local_y, *local_z;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    Read_n(&n, &local_n, my_rank, comm_sz, comm);
    Allocate_vectors(&local_x, &local_y, &local_z, local_n, comm);

    Read_vector(local_x, local_n, n, "x", my_rank, comm);
    Print_vector(local_x, local_n, n, "x is", my_rank, comm);
    Read_vector(local_y, local_n, n, "y", my_rank, comm);
    Print_vector(local_y, local_n, n, "y is", my_rank, comm);

    Parallel_vector_sum(local_x, local_y, local_z, local_n);
    Print_vector(local_z, local_n, n, "The sum is", my_rank, comm);

    free(local_x); free(local_y); free(local_z);
    MPI_Finalize();
    return 0;
}

/* MODIFICAÇÃO PRINCIPAL: Read_vector com MPI_Type_contiguous */
void Read_vector(double local_a[], int local_n, int n, char vec_name[],
                 int my_rank, MPI_Comm comm) {
    double* a = NULL;
    int local_ok = 1;

    /* Criar tipo contíguo de local_n doubles */
    MPI_Datatype contig_type;
    MPI_Type_contiguous(local_n, MPI_DOUBLE, &contig_type);
    MPI_Type_commit(&contig_type);

    if (my_rank == 0) {
        a = malloc(n * sizeof(double));
        if (a == NULL) local_ok = 0;
        Check_for_error(local_ok, "Read_vector", "malloc falhou", comm);
        printf("Entre com o vetor %s\n", vec_name);
        for (int i = 0; i < n; i++) scanf("%lf", &a[i]);
        /* count=1 porque o tipo já engloba local_n elementos */
        MPI_Scatter(a, 1, contig_type, local_a, 1, contig_type, 0, comm);
        free(a);
    } else {
        Check_for_error(local_ok, "Read_vector", "malloc falhou", comm);
        MPI_Scatter(a, 1, contig_type, local_a, 1, contig_type, 0, comm);
    }

    MPI_Type_free(&contig_type);
}

/* MODIFICAÇÃO PRINCIPAL: Print_vector com MPI_Type_contiguous */
void Print_vector(double local_b[], int local_n, int n, char title[],
                  int my_rank, MPI_Comm comm) {
    double* b = NULL;
    int local_ok = 1;

    MPI_Datatype contig_type;
    MPI_Type_contiguous(local_n, MPI_DOUBLE, &contig_type);
    MPI_Type_commit(&contig_type);

    if (my_rank == 0) {
        b = malloc(n * sizeof(double));
        if (b == NULL) local_ok = 0;
        Check_for_error(local_ok, "Print_vector", "malloc falhou", comm);
        /* count=1 porque o tipo já engloba local_n elementos */
        MPI_Gather(local_b, 1, contig_type, b, 1, contig_type, 0, comm);
        printf("%s\n", title);
        for (int i = 0; i < n; i++) printf("%f ", b[i]);
        printf("\n");
        free(b);
    } else {
        Check_for_error(local_ok, "Print_vector", "malloc falhou", comm);
        MPI_Gather(local_b, 1, contig_type, b, 1, contig_type, 0, comm);
    }

    MPI_Type_free(&contig_type);
}

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm) {
    int ok;
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
    if (ok == 0) {
        int my_rank; MPI_Comm_rank(comm, &my_rank);
        if (my_rank == 0) fprintf(stderr, "Erro em %s: %s\n", fname, message);
        MPI_Finalize(); exit(-1);
    }
}

void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm) {
    if (my_rank == 0) { printf("Ordem dos vetores? "); scanf("%d", n_p); }
    MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
    if (*n_p <= 0 || *n_p % comm_sz != 0) {
        Check_for_error(0, "Read_n", "n deve ser > 0 e divisível por comm_sz", comm);
    }
    *local_n_p = *n_p / comm_sz;
}

void Allocate_vectors(double** lx, double** ly, double** lz, int local_n, MPI_Comm comm) {
    *lx = malloc(local_n * sizeof(double));
    *ly = malloc(local_n * sizeof(double));
    *lz = malloc(local_n * sizeof(double));
    if (!*lx || !*ly || !*lz)
        Check_for_error(0, "Allocate_vectors", "malloc falhou", comm);
    else
        Check_for_error(1, "Allocate_vectors", "", comm);
}

void Parallel_vector_sum(double lx[], double ly[], double lz[], int local_n) {
    for (int i = 0; i < local_n; i++) lz[i] = lx[i] + ly[i];
}
