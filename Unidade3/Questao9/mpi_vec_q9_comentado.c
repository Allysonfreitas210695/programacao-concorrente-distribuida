/*
 * Questão 9 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Modificar as funções Read_vector e Print_vector do programa
 * mpi_vector_add.c original para usar MPI_Type_contiguous.
 *
 * O que é MPI_Type_contiguous?
 *   Cria um novo tipo derivado que representa 'count' elementos contíguos
 *   de um tipo base. Permite enviar/receber um bloco de elementos usando
 *   count=1 na chamada MPI_Scatter/MPI_Gather, em vez de count=local_n.
 *
 * Vantagem: encapsula o "quantos elementos" dentro do tipo, tornando
 * as chamadas de comunicação mais abstratas e flexíveis.
 *
 * Compile: mpicc -g -Wall -o mpi_vec_q9 mpi_vec_q9.c
 * Run:     mpiexec -n <p> ./mpi_vec_q9
 */

#include <stdio.h>    /* printf, scanf, fprintf */
#include <stdlib.h>   /* malloc, free, exit */
#include <mpi.h>

/* Declarações das funções */
void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_vectors(double** lx, double** ly, double** lz, int local_n, MPI_Comm comm);
void Read_vector(double local_a[], int local_n, int n, char vec_name[], int my_rank, MPI_Comm comm);
void Print_vector(double local_b[], int local_n, int n, char title[], int my_rank, MPI_Comm comm);
void Parallel_vector_sum(double lx[], double ly[], double lz[], int local_n);

int main(void) {
    int n, local_n, comm_sz, my_rank;
    double *local_x, *local_y, *local_z;   /* fatias locais dos três vetores */
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    /* Lê n global e calcula local_n */
    Read_n(&n, &local_n, my_rank, comm_sz, comm);

    /* Aloca os três vetores locais com verificação de erro */
    Allocate_vectors(&local_x, &local_y, &local_z, local_n, comm);

    /* Lê e imprime vetor x */
    Read_vector(local_x, local_n, n, "x", my_rank, comm);
    Print_vector(local_x, local_n, n, "x is", my_rank, comm);

    /* Lê e imprime vetor y */
    Read_vector(local_y, local_n, n, "y", my_rank, comm);
    Print_vector(local_y, local_n, n, "y is", my_rank, comm);

    /* Soma paralela: local_z = local_x + local_y */
    Parallel_vector_sum(local_x, local_y, local_z, local_n);

    /* Imprime o resultado da soma */
    Print_vector(local_z, local_n, n, "The sum is", my_rank, comm);

    free(local_x); free(local_y); free(local_z);
    MPI_Finalize();
    return 0;
}

/*
 * Read_vector — MODIFICAÇÃO PRINCIPAL
 *
 * Cria um MPI_Type_contiguous de local_n doubles, depois usa count=1
 * nesse tipo para o MPI_Scatter (ao invés de count=local_n em MPI_DOUBLE).
 *
 * Isso significa: "envia 1 bloco do tipo contig_type para cada processo",
 * onde contig_type já representa local_n doubles contíguos.
 */
void Read_vector(double local_a[], int local_n, int n, char vec_name[],
                 int my_rank, MPI_Comm comm) {
    double* a = NULL;
    int local_ok = 1;   /* flag de erro local (1 = ok, 0 = erro) */

    /* ─── Criação do tipo derivado ─── */
    MPI_Datatype contig_type;

    /*
     * MPI_Type_contiguous: define um bloco de 'local_n' elementos consecutivos
     * do tipo MPI_DOUBLE como um novo tipo chamado 'contig_type'.
     *
     * Parâmetros:
     *   local_n      → quantos elementos consecutivos compõem o novo tipo
     *   MPI_DOUBLE   → tipo base de cada elemento
     *   &contig_type → handle do novo tipo criado
     */
    MPI_Type_contiguous(local_n, MPI_DOUBLE, &contig_type);

    /*
     * MPI_Type_commit: registra o tipo derivado para uso em comunicações.
     * Obrigatório antes de usar o tipo em Send/Recv/Scatter/etc.
     */
    MPI_Type_commit(&contig_type);

    if (my_rank == 0) {
        /* Aloca vetor global somente no processo 0 */
        a = malloc(n * sizeof(double));
        if (a == NULL) local_ok = 0;

        /* Verifica se todos os processos estão ok (error check coletivo) */
        Check_for_error(local_ok, "Read_vector", "malloc falhou", comm);

        printf("Entre com o vetor %s\n", vec_name);
        for (int i = 0; i < n; i++) scanf("%lf", &a[i]);

        /*
         * MPI_Scatter com count=1 e tipo contig_type:
         *   "Envia 1 bloco de contig_type (= local_n doubles) para cada processo."
         *
         * Processo 0: a[0..local_n-1]
         * Processo 1: a[local_n..2*local_n-1]
         * ...
         *
         * Funcionalmente equivalente a:
         *   MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, ...)
         * mas encapsula o tamanho no tipo.
         */
        MPI_Scatter(a, 1, contig_type, local_a, 1, contig_type, 0, comm);

        free(a);
    } else {
        /* Processos não-raiz também participam do check coletivo */
        Check_for_error(local_ok, "Read_vector", "malloc falhou", comm);

        /* O primeiro argumento (a) não é usado em processos não-raiz,
         * mas a chamada MPI_Scatter DEVE ser feita por TODOS os processos
         * pois é uma operação coletiva. */
        MPI_Scatter(a, 1, contig_type, local_a, 1, contig_type, 0, comm);
    }

    /* Libera o tipo derivado — boa prática após uso */
    MPI_Type_free(&contig_type);
}

/*
 * Print_vector — MODIFICAÇÃO PRINCIPAL
 *
 * Mesma lógica do Read_vector, mas usa MPI_Gather para coletar as fatias
 * e imprimí-las no processo 0.
 */
void Print_vector(double local_b[], int local_n, int n, char title[],
                  int my_rank, MPI_Comm comm) {
    double* b = NULL;
    int local_ok = 1;

    /* Cria e registra o mesmo tipo contíguo */
    MPI_Datatype contig_type;
    MPI_Type_contiguous(local_n, MPI_DOUBLE, &contig_type);
    MPI_Type_commit(&contig_type);

    if (my_rank == 0) {
        b = malloc(n * sizeof(double));
        if (b == NULL) local_ok = 0;
        Check_for_error(local_ok, "Print_vector", "malloc falhou", comm);

        /*
         * MPI_Gather com count=1 e tipo contig_type:
         *   Processo 0 coleta 1 bloco de contig_type de cada processo
         *   e os concatena em b[].
         *
         * Equivalente a:
         *   MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, ...)
         */
        MPI_Gather(local_b, 1, contig_type, b, 1, contig_type, 0, comm);

        /* Imprime o vetor global reconstruído */
        printf("%s\n", title);
        for (int i = 0; i < n; i++) printf("%f ", b[i]);
        printf("\n");
        free(b);
    } else {
        Check_for_error(local_ok, "Print_vector", "malloc falhou", comm);
        /* Todos os processos devem chamar MPI_Gather (operação coletiva) */
        MPI_Gather(local_b, 1, contig_type, b, 1, contig_type, 0, comm);
    }

    MPI_Type_free(&contig_type);
}

/*
 * Check_for_error: verifica coletivamente se algum processo teve erro.
 * Usa MPI_Allreduce com MPI_MIN: se qualquer local_ok for 0, ok será 0.
 */
void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm) {
    int ok;
    /* MPI_MIN: resultado é 0 se QUALQUER processo passou 0 */
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
    if (ok == 0) {
        int my_rank; MPI_Comm_rank(comm, &my_rank);
        if (my_rank == 0) fprintf(stderr, "Erro em %s: %s\n", fname, message);
        MPI_Finalize(); exit(-1);
    }
}

/* Lê n do processo 0, faz broadcast, calcula local_n */
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm) {
    if (my_rank == 0) { printf("Ordem dos vetores? "); scanf("%d", n_p); }
    MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
    /* n deve ser positivo e divisível por comm_sz para MPI_Type_contiguous uniforme */
    if (*n_p <= 0 || *n_p % comm_sz != 0) {
        Check_for_error(0, "Read_n", "n deve ser > 0 e divisível por comm_sz", comm);
    }
    *local_n_p = *n_p / comm_sz;
}

/* Aloca três vetores locais e verifica sucesso coletivamente */
void Allocate_vectors(double** lx, double** ly, double** lz, int local_n, MPI_Comm comm) {
    *lx = malloc(local_n * sizeof(double));
    *ly = malloc(local_n * sizeof(double));
    *lz = malloc(local_n * sizeof(double));
    if (!*lx || !*ly || !*lz)
        Check_for_error(0, "Allocate_vectors", "malloc falhou", comm);
    else
        Check_for_error(1, "Allocate_vectors", "", comm);
}

/* Soma element-wise das fatias locais: lz[i] = lx[i] + ly[i] */
void Parallel_vector_sum(double lx[], double ly[], double lz[], int local_n) {
    for (int i = 0; i < local_n; i++) lz[i] = lx[i] + ly[i];
}
