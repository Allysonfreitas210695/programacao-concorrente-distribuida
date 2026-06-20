/* Questão 10 - Lista MPI
 * Usa MPI_Type_indexed para criar tipo da triangular superior
 * de uma matriz n×n armazenada como vetor 1D (row-major).
 * Processo 0 envia a triangular superior, processo 1 recebe e imprime.
 *
 * Compile: mpicc -g -Wall -o mpi_upper_q10 mpi_upper_q10.c
 * Run:     mpiexec -n 2 ./mpi_upper_q10
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(void) {
    int my_rank, comm_sz;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int n;
    int *matrix = NULL;

    if (my_rank == 0) {
        printf("Digite n (ordem da matriz): ");
        scanf("%d", &n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Conta elementos da triangular superior (incluindo diagonal): n*(n+1)/2 */
    int upper_count = n * (n + 1) / 2;

    /* Aloca arrays para MPI_Type_indexed */
    int *blocklengths   = malloc(n * sizeof(int));
    int *displacements  = malloc(n * sizeof(int));

    /* Linha i da triangular superior começa na coluna i e tem (n-i) elementos */
    for (int i = 0; i < n; i++) {
        blocklengths[i]  = n - i;              /* elementos na linha i acima da diagonal */
        displacements[i] = i * n + i;          /* offset em unidades de MPI_INT (não bytes) */
    }

    MPI_Datatype upper_tri_type;
    MPI_Type_indexed(n, blocklengths, displacements, MPI_INT, &upper_tri_type);
    MPI_Type_commit(&upper_tri_type);

    if (my_rank == 0) {
        /* Lê a matriz n×n */
        matrix = malloc(n * n * sizeof(int));
        printf("Digite os %d×%d = %d elementos (row-major):\n", n, n, n*n);
        for (int i = 0; i < n * n; i++) scanf("%d", &matrix[i]);

        /* Envia a triangular superior com UMA única chamada MPI_Send */
        MPI_Send(matrix, 1, upper_tri_type, 1, 0, MPI_COMM_WORLD);
        printf("Processo 0: triangular superior enviada.\n");
        free(matrix);

    } else if (my_rank == 1) {
        /* Recebe e armazena em vetor plano */
        int *recv_buf = malloc(upper_count * sizeof(int));

        /* Recebe com UMA única chamada MPI_Recv como vetor contíguo */
        MPI_Recv(recv_buf, upper_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Processo 1 recebeu triangular superior (%d elementos):\n", upper_count);
        for (int i = 0; i < upper_count; i++)
            printf("%d ", recv_buf[i]);
        printf("\n");
        free(recv_buf);
    }

    MPI_Type_free(&upper_tri_type);
    free(blocklengths);
    free(displacements);
    MPI_Finalize();
    return 0;
}
