/* Questão 14 - Lista MPI
 * Modificação de mpi_odd_even.c: Merge_low e Merge_high trocam ponteiros
 * em vez de copiar elementos.
 *
 * Trecho principal: apenas as funções Merge_low e Merge_high modificadas,
 * além de Sort (que agora gerencia os ponteiros).
 *
 * A ideia: em vez de copiar os menores/maiores elementos de volta para local_A,
 * simplesmente troca o ponteiro local_A com temp_C (que já contém o resultado).
 *
 * Compile: mpicc -g -Wall -O2 -o mpi_odd_even_q14 mpi_odd_even_q14.c
 * Run:     mpiexec -n <p> ./mpi_odd_even_q14 g <n>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

const int RMAX = 100;

void Usage(char* program);
void Print_list(int local_A[], int local_n, int rank);
void Merge_low(int** local_A_pp, int temp_B[], int** free_buf_pp, int local_n);
void Merge_high(int** local_A_pp, int temp_B[], int** free_buf_pp, int local_n);
void Generate_list(int local_A[], int local_n, int my_rank);
int  Compare(const void* a_p, const void* b_p);
void Get_args(int argc, char* argv[], int* global_n_p, int* local_n_p,
              char* gi_p, int my_rank, int p, MPI_Comm comm);
void Sort(int** local_A_pp, int local_n, int my_rank, int p, MPI_Comm comm);
void Odd_even_iter(int** local_A_pp, int temp_B[], int** free_buf_pp,
                   int local_n, int phase, int even_partner, int odd_partner,
                   int my_rank, int p, MPI_Comm comm);
void Print_global_list(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm);
void Read_list(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm);

int main(int argc, char* argv[]) {
    int my_rank, p;
    char g_i;
    int *local_A;
    int global_n, local_n;
    MPI_Comm comm;
    double start, finish;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    Get_args(argc, argv, &global_n, &local_n, &g_i, my_rank, p, comm);
    local_A = malloc(local_n * sizeof(int));

    if (g_i == 'g') {
        Generate_list(local_A, local_n, my_rank);
    } else {
        Read_list(local_A, local_n, my_rank, p, comm);
    }

    MPI_Barrier(comm);
    start = MPI_Wtime();

    Sort(&local_A, local_n, my_rank, p, comm);

    finish = MPI_Wtime();

    Print_global_list(local_A, local_n, my_rank, p, comm);

    if (my_rank == 0)
        printf("Tempo: %.6f s  (p=%d, n=%d)\n", finish - start, p, global_n);

    free(local_A);
    MPI_Finalize();
    return 0;
}

/* MODIFICAÇÃO: Merge_low agora troca ponteiros.
 * Escreve os local_n menores em *free_buf_pp, depois troca com *local_A_pp.
 * O buffer antigo de *local_A_pp vira o novo "livre" para a próxima fase,
 * eliminando aliasing e memory leak. */
void Merge_low(int** local_A_pp, int temp_B[], int** free_buf_pp, int local_n) {
    int* dst = *free_buf_pp;
    int ai = 0, bi = 0, ci = 0;
    while (ci < local_n) {
        if (ai < local_n && (bi >= local_n || (*local_A_pp)[ai] <= temp_B[bi]))
            dst[ci++] = (*local_A_pp)[ai++];
        else
            dst[ci++] = temp_B[bi++];
    }
    int* tmp    = *local_A_pp;
    *local_A_pp  = dst;
    *free_buf_pp = tmp;
}

/* MODIFICAÇÃO: Merge_high agora troca ponteiros. */
void Merge_high(int** local_A_pp, int temp_B[], int** free_buf_pp, int local_n) {
    int* dst = *free_buf_pp;
    int ai = local_n - 1, bi = local_n - 1, ci = local_n - 1;
    while (ci >= 0) {
        if (ai >= 0 && (bi < 0 || (*local_A_pp)[ai] >= temp_B[bi]))
            dst[ci--] = (*local_A_pp)[ai--];
        else
            dst[ci--] = temp_B[bi--];
    }
    int* tmp    = *local_A_pp;
    *local_A_pp  = dst;
    *free_buf_pp = tmp;
}

/* Sort adaptado: dois buffers alternantes — local_A (resultado atual)
 * e free_buf (disponível para escrita). Sem aliasing, sem leak. */
void Sort(int** local_A_pp, int local_n, int my_rank, int p, MPI_Comm comm) {
    int* temp_B  = malloc(local_n * sizeof(int));
    int* free_buf = malloc(local_n * sizeof(int));

    qsort(*local_A_pp, local_n, sizeof(int), Compare);

    int even_partner = (my_rank % 2 == 0) ? my_rank + 1 : my_rank - 1;
    int odd_partner  = (my_rank % 2 == 0) ? my_rank - 1 : my_rank + 1;
    if (even_partner < 0 || even_partner >= p) even_partner = MPI_PROC_NULL;
    if (odd_partner  < 0 || odd_partner  >= p) odd_partner  = MPI_PROC_NULL;

    for (int phase = 0; phase < p; phase++)
        Odd_even_iter(local_A_pp, temp_B, &free_buf, local_n,
                      phase, even_partner, odd_partner, my_rank, p, comm);

    free(temp_B);
    free(free_buf);  /* free_buf é sempre o buffer que NÃO tem o resultado */
}

void Odd_even_iter(int** local_A_pp, int temp_B[], int** free_buf_pp,
                   int local_n, int phase, int even_partner, int odd_partner,
                   int my_rank, int p, MPI_Comm comm) {
    int partner;
    if (phase % 2 == 0) partner = even_partner;
    else                 partner = odd_partner;

    if (partner == MPI_PROC_NULL) return;

    MPI_Sendrecv(*local_A_pp, local_n, MPI_INT, partner, 0,
                 temp_B,      local_n, MPI_INT, partner, 0,
                 comm, MPI_STATUS_IGNORE);

    if (my_rank < partner)
        Merge_low(local_A_pp, temp_B, free_buf_pp, local_n);
    else
        Merge_high(local_A_pp, temp_B, free_buf_pp, local_n);
}

/* Funções auxiliares */
void Generate_list(int local_A[], int local_n, int my_rank) {
    srand(my_rank + 1);
    for (int i = 0; i < local_n; i++) local_A[i] = rand() % RMAX;
}

int Compare(const void* a_p, const void* b_p) {
    return (*(int*)a_p - *(int*)b_p);
}

void Get_args(int argc, char* argv[], int* global_n_p, int* local_n_p,
              char* gi_p, int my_rank, int p, MPI_Comm comm) {
    if (my_rank == 0) {
        if (argc != 3) { Usage(argv[0]); MPI_Abort(comm, 1); }
        *gi_p      = argv[1][0];
        *global_n_p = atoi(argv[2]);
    }
    MPI_Bcast(gi_p,       1, MPI_CHAR, 0, comm);
    MPI_Bcast(global_n_p, 1, MPI_INT,  0, comm);
    *local_n_p = *global_n_p / p;
}

void Usage(char* program) {
    fprintf(stderr, "Uso: %s <g|i> <n>\n", program);
}

void Print_global_list(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm) {
    int* global_A = NULL;
    if (my_rank == 0) global_A = malloc(p * local_n * sizeof(int));
    MPI_Gather(local_A, local_n, MPI_INT, global_A, local_n, MPI_INT, 0, comm);
    if (my_rank == 0) {
        printf("Lista ordenada: ");
        for (int i = 0; i < p * local_n && i < 20; i++) printf("%d ", global_A[i]);
        if (p * local_n > 20) printf("...");
        printf("\n");
        free(global_A);
    }
}

void Read_list(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm) {
    int* global_A = NULL;
    if (my_rank == 0) {
        global_A = malloc(p * local_n * sizeof(int));
        printf("Digite %d inteiros:\n", p * local_n);
        for (int i = 0; i < p * local_n; i++) scanf("%d", &global_A[i]);
    }
    MPI_Scatter(global_A, local_n, MPI_INT, local_A, local_n, MPI_INT, 0, comm);
    if (my_rank == 0) free(global_A);
}
