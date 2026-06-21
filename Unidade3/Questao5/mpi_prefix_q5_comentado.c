/*
 * Questão 5 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Implementar somas de prefixos de três formas:
 *   (a) Serial — O(n), para referência conceitual.
 *   (b) Paralela SEM MPI_Scan — algoritmo de varredura em log2(p) fases.
 *   (c) Paralela COM MPI_Scan — usa a operação coletiva nativa do MPI.
 *
 * Definição: soma de prefixos do vetor [x0, x1, ..., x_{p-1}] é o vetor
 *   [x0, x0+x1, x0+x1+x2, ..., x0+...+x_{p-1}]
 *
 * Neste programa cada processo possui UM único valor (my_val).
 * Ao final, processo q deve ter a soma x[0] + x[1] + ... + x[q].
 *
 * Compile: mpicc -g -Wall -o mpi_prefix_q5 mpi_prefix_q5.c
 * Run:     mpiexec -n <p> ./mpi_prefix_q5   (p deve ser potência de 2 para (b))
 */

#include <stdio.h>    /* printf, fflush */
#include <stdlib.h>   /* rand, srand */
#include <mpi.h>

/*
 * (a) Função SERIAL de somas de prefixos — complexidade O(n).
 *
 * Recebe vetor x[0..n-1] e preenche prefix[i] = x[0] + x[1] + ... + x[i].
 * prefix[0] = x[0]             (caso base)
 * prefix[i] = prefix[i-1] + x[i]  (recorrência)
 *
 * Não é chamada no main paralelo, mas ilustra o conceito que os algoritmos
 * paralelos abaixo implementam de forma distribuída.
 */
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

    /* Cada processo gera seu valor com semente diferente (my_rank + 42).
     * Usar semente diferente garante valores distintos por processo. */
    srand(my_rank + 42);
    double my_val = (double)(rand() % 100);   /* valor entre 0 e 99 */

    /* ═══════════════════════════════════════════════════════════════
     * (b) ALGORITMO PARALELO SEM MPI_Scan
     *
     * Ideia: varredura em log2(p) fases com 'half' dobrando a cada passo.
     *
     * Na fase com 'half':
     *   - Processo q envia  seu acumulador atual para o processo q+half.
     *   - Processo q recebe o acumulador do processo q-half.
     *   - q acumula: cur += recebido.
     *
     * Após a fase half=1:  cur[q] = x[q-1] + x[q]        (soma com vizinho)
     * Após a fase half=2:  cur[q] = x[q-3]+...+x[q]       (soma com 4 vizinhos)
     * ...
     * Após a fase half=p/2: cur[q] = x[0]+...+x[q]   ← resultado correto
     *
     * Complexidade: O(log p) fases, cada uma com 1 Send e 1 Recv.
     * ═══════════════════════════════════════════════════════════════ */
    double cur = my_val;   /* acumulador inicializado com o valor local */

    for (int half = 1; half < comm_sz; half <<= 1) {   /* half = 1, 2, 4, 8, ... */
        double received = 0.0;

        /* Envia o acumulador atual para o processo à direita (q + half).
         * Só envia se o destinatário existe (rank < comm_sz). */
        if (my_rank + half < comm_sz)
            MPI_Send(&cur, 1, MPI_DOUBLE, my_rank + half, 0, MPI_COMM_WORLD);

        /* Recebe o acumulador do processo à esquerda (q - half).
         * Só recebe se a fonte existe (rank >= 0). */
        if (my_rank - half >= 0)
            MPI_Recv(&received, 1, MPI_DOUBLE, my_rank - half, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Acumula: agora 'cur' inclui a contribuição dos processos mais à esquerda */
        cur += received;

        /* Barreira garante que todos terminaram esta fase antes da próxima.
         * Sem isso, um processo poderia começar a fase seguinte com dados
         * desatualizados do buffer de recepção. */
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double prefix_val = cur;   /* resultado final da soma de prefixo */

    /* ── Impressão em ordem usando token ring ── */
    if (my_rank == 0) {
        printf("=== (b) Paralelo sem MPI_Scan ===\n");
        printf("Proc %d: valor=%.0f, prefix_sum=%.0f\n", my_rank, my_val, prefix_val);
        fflush(stdout);
        /* Envia token para o processo 1 se houver mais processos */
        if (comm_sz > 1) { int t = 1; MPI_Send(&t, 1, MPI_INT, 1, 99, MPI_COMM_WORLD); }
    } else {
        int t;
        /* Aguarda o token do processo anterior antes de imprimir */
        MPI_Recv(&t, 1, MPI_INT, my_rank - 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Proc %d: valor=%.0f, prefix_sum=%.0f\n", my_rank, my_val, prefix_val);
        fflush(stdout);
        /* Repassa o token para o próximo processo */
        if (my_rank < comm_sz - 1)
            MPI_Send(&t, 1, MPI_INT, my_rank + 1, 99, MPI_COMM_WORLD);
    }

    /* ═══════════════════════════════════════════════════════════════
     * (c) ALGORITMO COM MPI_Scan
     *
     * MPI_Scan é uma operação coletiva que realiza a soma de prefixos
     * automaticamente, de forma eficiente e implementada internamente pelo MPI.
     *
     * Após MPI_Scan:
     *   scan_result no processo q = my_val[0] + my_val[1] + ... + my_val[q]
     *
     * Parâmetros:
     *   &my_val       → valor de entrada de cada processo
     *   &scan_result  → resultado da soma de prefixo neste processo
     *   1             → número de elementos
     *   MPI_DOUBLE    → tipo
     *   MPI_SUM       → operação de redução
     *   MPI_COMM_WORLD
     * ═══════════════════════════════════════════════════════════════ */
    double scan_result;
    MPI_Scan(&my_val, &scan_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Barreira para garantir que todos terminaram o Scan antes de imprimir */
    MPI_Barrier(MPI_COMM_WORLD);

    /* ── Impressão em ordem usando token ring (tag 98 para não confundir com (b)) ── */
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

/*
 * EXEMPLO com 4 processos e valores [10, 20, 30, 40]:
 *
 * (b) Algoritmo manual:
 *   Fase half=1: proc0 envia 10→1; proc1 recebe 10, cur[1]=30
 *                proc1 envia 20→2; proc2 recebe 20, cur[2]=50
 *                proc2 envia 30→3; proc3 recebe 30, cur[3]=70
 *   Fase half=2: proc0 envia 10→2; proc2 recebe 10, cur[2]=60
 *                proc1 envia 30→3; proc3 recebe 30, cur[3]=100
 *   Resultado: [10, 30, 60, 100] ← correto!
 *
 * (c) MPI_Scan produz [10, 30, 60, 100] diretamente.
 */
