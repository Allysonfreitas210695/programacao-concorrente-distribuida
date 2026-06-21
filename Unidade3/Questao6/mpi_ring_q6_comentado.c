/*
 * Questão 6 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Implementar duas operações coletivas usando passagem de mensagens
 * em ANEL (ring), sem usar as operações coletivas prontas do MPI:
 *   (a) Allreduce (soma global) via anel.
 *   (b) Somas de prefixos via anel.
 *
 * Conceito de anel:
 *   Cada processo q envia para (q+1) % p e recebe de (q-1+p) % p.
 *   Após p-1 passos, cada valor percorreu todos os processos.
 *
 * Compile: mpicc -g -Wall -o mpi_ring_q6 mpi_ring_q6.c
 * Run:     mpiexec -n <p> ./mpi_ring_q6
 */

#include <stdio.h>    /* printf, fflush */
#include <stdlib.h>   /* rand, srand */
#include <mpi.h>

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    MPI_Status status;   /* informações sobre mensagem recebida */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Cada processo gera um valor aleatório entre 0 e 19 */
    srand(my_rank + 10);
    int my_val = rand() % 20;

    /* Vizinhos no anel:
     *   dest   = processo para quem ENVIO (próximo no anel)
     *   source = processo de quem RECEBO (anterior no anel)
     * O módulo garante que o anel "fecha": proc p-1 envia para proc 0. */
    int dest   = (my_rank + 1) % comm_sz;
    int source = (my_rank - 1 + comm_sz) % comm_sz;
    int sendtag = 0, recvtag = 0;   /* tags das mensagens */

    /* ═══════════════════════════════════════════════════════════════
     * (a) ALLREDUCE via anel
     *
     * Cada processo circula seu valor pelo anel. Após p-1 iterações,
     * cada processo acumulou todos os valores → tem a soma global.
     *
     * Iteração i: cada processo passa adiante o valor que recebeu
     *             na iteração anterior (temp_val).
     * ═══════════════════════════════════════════════════════════════ */
    int sum = my_val;       /* acumulador da soma total */
    int temp_val = my_val;  /* valor que circula pelo anel neste passo */

    for (int i = 1; i < comm_sz; i++) {
        /*
         * MPI_Sendrecv_replace: envia E recebe usando o MESMO buffer.
         *   Envia temp_val para 'dest' e sobrescreve temp_val com o dado
         *   recebido de 'source'. Isso evita a necessidade de dois buffers.
         *
         * Parâmetros:
         *   &temp_val     → buffer único para envio e recepção
         *   1             → número de elementos
         *   MPI_INT       → tipo
         *   dest          → destinatário do envio
         *   sendtag       → tag para o envio
         *   source        → fonte da recepção
         *   recvtag       → tag para a recepção
         *   MPI_COMM_WORLD
         *   &status       → informações da mensagem recebida
         */
        MPI_Sendrecv_replace(&temp_val, 1, MPI_INT,
                             dest,   sendtag,
                             source, recvtag,
                             MPI_COMM_WORLD, &status);

        /* Acumula o valor recebido (que pertencia a outro processo) */
        sum += temp_val;
    }
    /* Após p-1 iterações: sum contém x[0]+x[1]+...+x[p-1] */

    printf("[Anel Allreduce] Proc %d: my_val=%d, sum=%d\n", my_rank, my_val, sum);
    fflush(stdout);

    /* Verificação cruzada com MPI_Allreduce para confirmar o resultado */
    int mpi_sum;
    MPI_Allreduce(&my_val, &mpi_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (my_rank == 0)
        printf("[MPI_Allreduce] sum=%d  (deve ser igual ao anel)\n\n", mpi_sum);

    /* Barreira garante que todos terminaram (a) antes de iniciar (b) */
    MPI_Barrier(MPI_COMM_WORLD);

    /* ═══════════════════════════════════════════════════════════════
     * (b) SOMAS DE PREFIXOS via anel
     *
     * Objetivo: processo q acumula somente os valores dos processos
     * com rank MENOR que q, ou seja: prefix[q] = x[0]+...+x[q].
     *
     * Estratégia: circulamos os valores pelo anel igual ao (a),
     * mas só somamos o valor recebido SE ele veio de um processo
     * com rank < my_rank.
     *
     * Na iteração i: temp_val contém o valor do processo (my_rank - i) mod p.
     * Se esse rank for menor que my_rank, somamos.
     * ═══════════════════════════════════════════════════════════════ */

    /* Reinicia temp_val com o valor original para nova circulação */
    temp_val = my_val;
    int prefix = my_val;   /* prefix começa com o próprio valor (x[q]) */

    for (int i = 1; i < comm_sz; i++) {
        /* Circula o valor pelo anel (igual ao allreduce) */
        MPI_Sendrecv_replace(&temp_val, 1, MPI_INT,
                             dest,   sendtag,
                             source, recvtag,
                             MPI_COMM_WORLD, &status);

        /* Calcula qual processo enviou originalmente este valor.
         * Na iteração i, o valor veio do processo (my_rank - i + p) % p.
         * Exemplo: my_rank=3, i=1 → veio do processo 2 (valor de x[2]).
         *          my_rank=3, i=2 → veio do processo 1 (valor de x[1]).
         *          my_rank=3, i=3 → veio do processo 0 (valor de x[0]).
         *          my_rank=3, i=4 → veio do processo 3 (o próprio!) → NÃO soma. */
        int sender_rank = (my_rank - i + comm_sz) % comm_sz;

        /* Só acumula se o remetente original tem rank menor — é um prefixo */
        if (sender_rank < my_rank)
            prefix += temp_val;
    }
    /* Resultado: prefix[q] = x[0] + x[1] + ... + x[q] */

    MPI_Barrier(MPI_COMM_WORLD);
    printf("[Anel PrefixSum] Proc %d: my_val=%d, prefix_sum=%d\n", my_rank, my_val, prefix);
    fflush(stdout);

    /* Verificação com MPI_Scan */
    int mpi_prefix;
    MPI_Scan(&my_val, &mpi_prefix, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[MPI_Scan]       Proc %d: prefix_sum=%d\n", my_rank, mpi_prefix);

    MPI_Finalize();
    return 0;
}

/*
 * EXEMPLO com 3 processos e valores [5, 10, 15]:
 *
 * (a) Allreduce via anel:
 *   i=1: cada um circula seu valor. Proc0 recebe 15 (de proc2), sum[0]=20
 *         Proc1 recebe 5  (de proc0), sum[1]=15
 *         Proc2 recebe 10 (de proc1), sum[2]=25
 *   i=2: circula novamente.
 *         Proc0 recebe 10, sum[0]=30; Proc1 recebe 15, sum[1]=30; Proc2 recebe 5, sum[2]=30
 *   Resultado: todos com sum=30 ✓
 *
 * (b) Prefix via anel (proc2, my_rank=2):
 *   i=1: recebe temp=10 de proc1 → sender_rank=(2-1)%3=1 < 2 → prefix=15+10=25
 *   i=2: recebe temp=5  de proc0 → sender_rank=(2-2)%3=0 < 2 → prefix=25+5=30
 *   Resultado: prefix[2]=30 = 5+10+15 ✓
 */
