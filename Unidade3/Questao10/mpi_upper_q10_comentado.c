/*
 * Questão 10 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Usar MPI_Type_indexed para criar um tipo derivado que represente
 * os elementos da TRIANGULAR SUPERIOR de uma matriz n×n armazenada como
 * vetor 1D (row-major / linha a linha).
 *
 * O que é MPI_Type_indexed?
 *   Cria um tipo derivado com blocos de tamanhos DIFERENTES em posições
 *   DIFERENTES dentro do buffer. Cada bloco é descrito por um par
 *   (blocklength, displacement).
 *
 * Por que usar aqui?
 *   A triangular superior tem linhas de comprimentos diferentes:
 *   linha 0: n elementos (colunas 0..n-1)
 *   linha 1: n-1 elementos (colunas 1..n-1)
 *   ...
 *   linha n-1: 1 elemento (coluna n-1)
 *   → blocos de tamanhos DIFERENTES, em offsets DIFERENTES → MPI_Type_indexed.
 *
 * Compile: mpicc -g -Wall -o mpi_upper_q10 mpi_upper_q10.c
 * Run:     mpiexec -n 2 ./mpi_upper_q10
 */

#include <stdio.h>    /* printf, scanf, fprintf */
#include <stdlib.h>   /* malloc, free */
#include <mpi.h>

int main(void) {
    int my_rank, comm_sz;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int n;
    int *matrix = NULL;

    /* Processo 0 lê n e distribui para o processo 1 */
    if (my_rank == 0) {
        printf("Digite n (ordem da matriz): ");
        scanf("%d", &n);
    }
    /* Broadcast de n: processo 1 também precisa saber n para alocar */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Número de elementos na triangular superior (diagonal incluída):
     * soma 1+2+...+n = n*(n+1)/2 */
    int upper_count = n * (n + 1) / 2;

    /* Arrays que descrevem os blocos do tipo indexado */
    int *blocklengths  = malloc(n * sizeof(int));   /* tamanho de cada bloco */
    int *displacements = malloc(n * sizeof(int));   /* posição de cada bloco no vetor */

    /*
     * Para a linha i da triangular superior (0 ≤ i < n):
     *   - Começa na coluna i (diagonal) e vai até a coluna n-1.
     *   - Portanto tem (n - i) elementos.
     *   - No vetor 1D row-major, o elemento [i][i] está na posição i*n + i.
     *
     * Exemplo com n=3:
     *   linha 0: blocklength=3, displacement=0   (elementos [0][0],[0][1],[0][2])
     *   linha 1: blocklength=2, displacement=4   (elementos [1][1],[1][2])
     *   linha 2: blocklength=1, displacement=8   (elemento  [2][2])
     */
    for (int i = 0; i < n; i++) {
        blocklengths[i]  = n - i;        /* número de elementos nesta linha */
        displacements[i] = i * n + i;    /* offset em unidades de MPI_INT (não bytes!) */
    }

    /* ─── Criação do tipo indexado ─── */
    MPI_Datatype upper_tri_type;

    /*
     * MPI_Type_indexed: cria tipo com n blocos de tamanhos e posições variados.
     *
     * Parâmetros:
     *   n               → número de blocos
     *   blocklengths    → array com o tamanho (em elementos) de cada bloco
     *   displacements   → array com o offset (em unidades do tipo base) de cada bloco
     *   MPI_INT         → tipo base de cada elemento
     *   &upper_tri_type → handle do novo tipo
     *
     * IMPORTANTE: displacements são em unidades de MPI_INT (não bytes).
     *   Para offsets em bytes, usaria MPI_Type_create_hindexed.
     */
    MPI_Type_indexed(n, blocklengths, displacements, MPI_INT, &upper_tri_type);

    /* Registra o tipo para uso em comunicações */
    MPI_Type_commit(&upper_tri_type);

    if (my_rank == 0) {
        /* Aloca e lê a matriz n×n completa */
        matrix = malloc(n * n * sizeof(int));
        printf("Digite os %d×%d = %d elementos (row-major):\n", n, n, n*n);
        for (int i = 0; i < n * n; i++) scanf("%d", &matrix[i]);

        /*
         * MPI_Send com count=1 e upper_tri_type:
         *   Envia exatamente os elementos da triangular superior.
         *   O tipo indexado determina QUAIS posições do buffer 'matrix' enviar.
         *
         * Uma única chamada MPI_Send transmite n*(n+1)/2 elementos
         * não contíguos no buffer, como se fossem um único dado.
         */
        MPI_Send(matrix, 1, upper_tri_type, 1, 0, MPI_COMM_WORLD);
        printf("Processo 0: triangular superior enviada.\n");
        free(matrix);

    } else if (my_rank == 1) {
        /* Aloca buffer plano para receber os upper_count elementos */
        int *recv_buf = malloc(upper_count * sizeof(int));

        /*
         * MPI_Recv com count=upper_count e MPI_INT:
         *   Recebe os elementos como vetor contíguo simples.
         *   O processo 0 enviou com o tipo indexado (não contíguo),
         *   mas o MPI serializa os dados; o receptor pode recebê-los
         *   em qualquer buffer contíguo do tamanho correto.
         *
         * Após o recebimento:
         *   recv_buf[0] = matrix[0][0]
         *   recv_buf[1] = matrix[0][1]
         *   ...
         *   recv_buf[upper_count-1] = matrix[n-1][n-1]
         */
        MPI_Recv(recv_buf, upper_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Processo 1 recebeu triangular superior (%d elementos):\n", upper_count);
        for (int i = 0; i < upper_count; i++)
            printf("%d ", recv_buf[i]);
        printf("\n");
        free(recv_buf);
    }

    /* Libera o tipo derivado */
    MPI_Type_free(&upper_tri_type);
    free(blocklengths);
    free(displacements);

    MPI_Finalize();
    return 0;
}

/*
 * EXEMPLO com n=3 (matriz armazenada como vetor de 9 ints):
 *
 *   matrix = [1, 2, 3, 4, 5, 6, 7, 8, 9]
 *   Representa:
 *     1 2 3
 *     4 5 6
 *     7 8 9
 *
 *   blocklengths  = [3, 2, 1]
 *   displacements = [0, 4, 8]
 *
 *   Tipo upper_tri_type captura: matrix[0,1,2], matrix[4,5], matrix[8]
 *   → elementos: 1, 2, 3, 5, 6, 9  (triangular superior)
 *
 *   recv_buf = [1, 2, 3, 5, 6, 9]
 */
