/*
 * Questão 2 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Modificar mpi_output.c para garantir que as mensagens sejam
 * impressas NA ORDEM dos ranks (0, 1, 2, ..., p-1).
 *
 * Problema original: cada processo imprimia sem coordenação, gerando
 * saída em ordem aleatória pois o scheduler do SO decide quem roda primeiro.
 *
 * Solução — token ring:
 *   Processo 0 imprime primeiro e passa um "token" (mensagem vazia) para 1.
 *   Processo k aguarda o token de k-1, imprime, e passa para k+1.
 *   Dessa forma a ordem de impressão segue obrigatoriamente o rank.
 *
 * Compile: mpicc -g -Wall -o mpi_output_q2 mpi_output_q2.c
 * Run:     mpiexec -n <p> ./mpi_output_q2
 */

#include <stdio.h>   /* printf, fflush */
#include <mpi.h>     /* MPI_Init, MPI_Send, MPI_Recv, etc. */

int main(void) {
    int my_rank, comm_sz;
    MPI_Status status;   /* struct com informações sobre a mensagem recebida */

    /* Inicializa o ambiente de execução MPI */
    MPI_Init(NULL, NULL);

    /* Obtém o número total de processos */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Obtém o rank do processo atual */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        /* ── PROCESSO 0: imprime e dispara o token ── */

        /* Imprime a mensagem deste processo */
        printf("Proc %d de %d > Does anyone have a toothpick?\n", my_rank, comm_sz);

        /* fflush garante que a mensagem vai ao terminal ANTES de continuar.
         * Sem isso, o buffer do stdio pode segurar a saída. */
        fflush(stdout);

        /* Só envia o token se existir pelo menos mais um processo */
        if (comm_sz > 1) {
            int token = 1;   /* conteúdo do token — qualquer valor serve */
            /*
             * MPI_Send: envia 1 inteiro para o processo de rank 1, tag 0.
             * Parâmetros:
             *   &token          → buffer de envio
             *   1               → quantidade de elementos
             *   MPI_INT         → tipo do dado
             *   1               → destino (próximo processo)
             *   0               → tag da mensagem (identificador arbitrário)
             *   MPI_COMM_WORLD  → comunicador
             */
            MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        }

    } else {
        /* ── PROCESSOS 1, 2, ..., p-1: aguardam o token, imprimem, repassam ── */

        int token;   /* variável que receberá o token do processo anterior */

        /*
         * MPI_Recv: bloqueia até receber 1 inteiro do processo (my_rank - 1), tag 0.
         * Parâmetros:
         *   &token          → buffer de recepção
         *   1               → quantidade máxima de elementos
         *   MPI_INT         → tipo esperado
         *   my_rank - 1     → fonte (processo imediatamente anterior)
         *   0               → tag esperada
         *   MPI_COMM_WORLD  → comunicador
         *   &status         → informações extras sobre a mensagem recebida
         *
         * Este bloqueio é o que GARANTE a ordem: este processo só avança
         * após o anterior ter terminado de imprimir e enviado o token.
         */
        MPI_Recv(&token, 1, MPI_INT, my_rank - 1, 0, MPI_COMM_WORLD, &status);

        /* Agora é seguro imprimir — todos os processos de rank menor já imprimiram */
        printf("Proc %d de %d > Does anyone have a toothpick?\n", my_rank, comm_sz);
        fflush(stdout);   /* força saída imediata para o terminal */

        /* Repassa o token para o próximo processo, a menos que seja o último */
        if (my_rank < comm_sz - 1) {
            MPI_Send(&token, 1, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD);
        }
        /* Se my_rank == comm_sz - 1: é o último processo, não há próximo. */
    }

    /* Finaliza o ambiente MPI */
    MPI_Finalize();
    return 0;
}

/*
 * RESUMO DO FLUXO DE MENSAGENS (exemplo com 4 processos):
 *
 *   Proc 0: imprime → Send(token → 1)
 *   Proc 1: Recv(token ← 0) → imprime → Send(token → 2)
 *   Proc 2: Recv(token ← 1) → imprime → Send(token → 3)
 *   Proc 3: Recv(token ← 2) → imprime → (fim)
 *
 * A cadeia de Recv bloqueia cada processo até que o anterior termine,
 * garantindo impressão em ordem estrita de rank.
 */
