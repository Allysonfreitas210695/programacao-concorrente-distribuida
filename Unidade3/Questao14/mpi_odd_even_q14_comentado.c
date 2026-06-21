/*
 * Questão 14 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Modificar o algoritmo de ordenação odd-even transposition sort
 * paralelo para que Merge_low e Merge_high TROQUEM PONTEIROS em vez de
 * copiar elementos de volta para local_A.
 *
 * Problema da versão original:
 *   Após o merge, os menores/maiores elementos eram copiados elemento a elemento
 *   de volta para local_A[] — O(local_n) cópias desnecessárias.
 *
 * Solução da Q14:
 *   Escreve o resultado em um buffer alternante (free_buf) e troca os ponteiros
 *   (*local_A_pp ↔ *free_buf_pp). Sem cópia — apenas troca de ponteiros O(1).
 *
 * Algoritmo Odd-Even Transposition Sort:
 *   Cada processo tem local_n elementos já ordenados localmente.
 *   Em p fases alternadas (par/ímpar), processos vizinhos trocam dados
 *   e mantêm os menores (Merge_low) ou maiores (Merge_high).
 *   Após p fases, a lista global está ordenada.
 *
 * Compile: mpicc -g -Wall -O2 -o mpi_odd_even_q14 mpi_odd_even_q14.c
 * Run:     mpiexec -n <p> ./mpi_odd_even_q14 g <n>
 *          (g = gerar aleatoriamente, i = ler do stdin; n = tamanho global)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

const int RMAX = 100;   /* valor máximo gerado aleatoriamente */

/* Protótipos */
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
    char g_i;        /* 'g' = gerar aleatório, 'i' = ler do usuário */
    int *local_A;    /* ponteiro para o array local (será trocado pelo Sort) */
    int global_n;    /* tamanho global do array */
    int local_n;     /* tamanho do array local = global_n / p */
    MPI_Comm comm;
    double start, finish;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    /* Obtém argumentos (g_i e global_n) da linha de comando */
    Get_args(argc, argv, &global_n, &local_n, &g_i, my_rank, p, comm);

    /* Aloca o array local com local_n elementos */
    local_A = malloc(local_n * sizeof(int));

    if (g_i == 'g') {
        /* Preenche com números aleatórios (semente baseada no rank) */
        Generate_list(local_A, local_n, my_rank);
    } else {
        /* Processo 0 lê do stdin e distribui com Scatter */
        Read_list(local_A, local_n, my_rank, p, comm);
    }

    /* Sincroniza antes de medir o tempo */
    MPI_Barrier(comm);
    start = MPI_Wtime();

    /*
     * Sort recebe um PONTEIRO PARA PONTEIRO (&local_A).
     * Isso é necessário porque Sort pode trocar o ponteiro local_A
     * com o buffer interno — se recebesse só int*, não poderia
     * modificar o ponteiro no main.
     */
    Sort(&local_A, local_n, my_rank, p, comm);

    finish = MPI_Wtime();

    /* Imprime até 20 elementos do resultado global */
    Print_global_list(local_A, local_n, my_rank, p, comm);

    if (my_rank == 0)
        printf("Tempo: %.6f s  (p=%d, n=%d)\n", finish - start, p, global_n);

    free(local_A);
    MPI_Finalize();
    return 0;
}

/*
 * Merge_low — MODIFICAÇÃO PRINCIPAL
 *
 * Problema original: após o merge, copiava os local_n menores elementos
 * de volta para local_A[] em um loop O(local_n).
 *
 * Solução Q14: escreve o resultado diretamente em *free_buf_pp (buffer disponível),
 * depois TROCA os ponteiros *local_A_pp e *free_buf_pp em O(1).
 *
 * Resultado: *local_A_pp aponta para o buffer com os menores elementos.
 *            *free_buf_pp aponta para o buffer antigo (agora disponível para reutilizar).
 *
 * Parâmetros:
 *   local_A_pp  → ponteiro para o ponteiro do array local (será redirecionado)
 *   temp_B[]    → array recebido do processo vizinho (local_n elementos)
 *   free_buf_pp → ponteiro para o buffer disponível para escrita
 *   local_n     → número de elementos em cada array
 */
void Merge_low(int** local_A_pp, int temp_B[], int** free_buf_pp, int local_n) {
    /* dst: onde os menores elementos serão escritos (buffer livre) */
    int* dst = *free_buf_pp;

    /* Índices para percorrer local_A (ai), temp_B (bi) e dst (ci) */
    int ai = 0, bi = 0, ci = 0;

    /* Merge clássico de dois arrays ordenados, pegando os local_n MENORES */
    while (ci < local_n) {
        /* Escolhe o menor entre local_A[ai] e temp_B[bi].
         * Prioriza local_A em caso de empate (<=). */
        if (ai < local_n && (bi >= local_n || (*local_A_pp)[ai] <= temp_B[bi]))
            dst[ci++] = (*local_A_pp)[ai++];
        else
            dst[ci++] = temp_B[bi++];
    }
    /* dst[] agora contém os local_n menores elementos */

    /* Troca de ponteiros em O(1) — sem cópia */
    int* tmp    = *local_A_pp;   /* salva ponteiro antigo de local_A */
    *local_A_pp  = dst;           /* local_A aponta para o resultado */
    *free_buf_pp = tmp;           /* buffer antigo fica disponível */
}

/*
 * Merge_high — MODIFICAÇÃO PRINCIPAL (simétrico ao Merge_low)
 *
 * Em vez dos menores, mantém os local_n MAIORES elementos.
 * Percorre os arrays de trás para frente.
 */
void Merge_high(int** local_A_pp, int temp_B[], int** free_buf_pp, int local_n) {
    int* dst = *free_buf_pp;

    /* Percorre de trás para frente para pegar os maiores */
    int ai = local_n - 1, bi = local_n - 1, ci = local_n - 1;

    while (ci >= 0) {
        /* Escolhe o maior entre local_A[ai] e temp_B[bi] */
        if (ai >= 0 && (bi < 0 || (*local_A_pp)[ai] >= temp_B[bi]))
            dst[ci--] = (*local_A_pp)[ai--];
        else
            dst[ci--] = temp_B[bi--];
    }

    /* Troca de ponteiros em O(1) */
    int* tmp    = *local_A_pp;
    *local_A_pp  = dst;
    *free_buf_pp = tmp;
}

/*
 * Sort: organiza o algoritmo odd-even com dois buffers alternantes.
 *
 * Dois buffers: *local_A_pp (resultado atual) e free_buf (disponível).
 * Merge_low/Merge_high escrevem em free_buf e trocam com *local_A_pp.
 * Nunca há aliasing (os dois ponteiros sempre apontam para áreas distintas).
 */
void Sort(int** local_A_pp, int local_n, int my_rank, int p, MPI_Comm comm) {
    /* temp_B: buffer para receber os dados do processo vizinho */
    int* temp_B  = malloc(local_n * sizeof(int));

    /* free_buf: buffer alternante para o merge sem cópia */
    int* free_buf = malloc(local_n * sizeof(int));

    /* Primeiro passo: ordenação local de cada processo com qsort */
    qsort(*local_A_pp, local_n, sizeof(int), Compare);

    /* Calcula os parceiros para fases pares e ímpares.
     * Fase par:  processo q troca com q+1 (se q par) ou q-1 (se q ímpar).
     * Fase ímpar: processo q troca com q-1 (se q par) ou q+1 (se q ímpar).
     * MPI_PROC_NULL: parceiro inexistente (borda do array de processos). */
    int even_partner = (my_rank % 2 == 0) ? my_rank + 1 : my_rank - 1;
    int odd_partner  = (my_rank % 2 == 0) ? my_rank - 1 : my_rank + 1;
    if (even_partner < 0 || even_partner >= p) even_partner = MPI_PROC_NULL;
    if (odd_partner  < 0 || odd_partner  >= p) odd_partner  = MPI_PROC_NULL;

    /* p fases de odd-even (prova de corretude: p fases são suficientes) */
    for (int phase = 0; phase < p; phase++)
        Odd_even_iter(local_A_pp, temp_B, &free_buf, local_n,
                      phase, even_partner, odd_partner, my_rank, p, comm);

    /* Libera os buffers auxiliares.
     * IMPORTANTE: free_buf sempre aponta para o buffer que NÃO contém o resultado
     * atual (a troca de ponteiros garante isso). */
    free(temp_B);
    free(free_buf);
}

/*
 * Odd_even_iter: uma fase do algoritmo.
 *   - Fase par:  usa even_partner.
 *   - Fase ímpar: usa odd_partner.
 *   - Troca arrays com o parceiro via MPI_Sendrecv.
 *   - Chama Merge_low (processo de rank menor) ou Merge_high (rank maior).
 */
void Odd_even_iter(int** local_A_pp, int temp_B[], int** free_buf_pp,
                   int local_n, int phase, int even_partner, int odd_partner,
                   int my_rank, int p, MPI_Comm comm) {
    int partner;
    /* Seleciona o parceiro com base na paridade da fase */
    if (phase % 2 == 0) partner = even_partner;
    else                 partner = odd_partner;

    /* MPI_PROC_NULL: parceiro inexistente → não faz nada nesta fase */
    if (partner == MPI_PROC_NULL) return;

    /*
     * MPI_Sendrecv: envia local_A para o parceiro e recebe em temp_B.
     * Operação simultânea evita deadlock (se usasse Send+Recv separados,
     * dois processos tentando enviar ao mesmo tempo travariam).
     *
     * Após esta chamada:
     *   temp_B[] = local_A do processo 'partner'
     */
    MPI_Sendrecv(*local_A_pp, local_n, MPI_INT, partner, 0,
                 temp_B,      local_n, MPI_INT, partner, 0,
                 comm, MPI_STATUS_IGNORE);

    if (my_rank < partner)
        /* Processo de rank menor fica com os menores elementos */
        Merge_low(local_A_pp, temp_B, free_buf_pp, local_n);
    else
        /* Processo de rank maior fica com os maiores elementos */
        Merge_high(local_A_pp, temp_B, free_buf_pp, local_n);
}

/* ─── Funções auxiliares ─── */

/* Gera lista aleatória com semente baseada no rank (reprodutível por processo) */
void Generate_list(int local_A[], int local_n, int my_rank) {
    srand(my_rank + 1);
    for (int i = 0; i < local_n; i++) local_A[i] = rand() % RMAX;
}

/* Comparador para qsort: ordena inteiros em ordem crescente */
int Compare(const void* a_p, const void* b_p) {
    return (*(int*)a_p - *(int*)b_p);
}

/* Processa argumentos da linha de comando e distribui via Bcast */
void Get_args(int argc, char* argv[], int* global_n_p, int* local_n_p,
              char* gi_p, int my_rank, int p, MPI_Comm comm) {
    if (my_rank == 0) {
        if (argc != 3) { Usage(argv[0]); MPI_Abort(comm, 1); }
        *gi_p       = argv[1][0];   /* 'g' ou 'i' */
        *global_n_p = atoi(argv[2]);
    }
    MPI_Bcast(gi_p,       1, MPI_CHAR, 0, comm);
    MPI_Bcast(global_n_p, 1, MPI_INT,  0, comm);
    *local_n_p = *global_n_p / p;   /* divisão exata exigida */
}

void Usage(char* program) {
    fprintf(stderr, "Uso: %s <g|i> <n>\n", program);
}

/* Reúne todos os arrays locais no processo 0 e imprime até 20 elementos */
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

/* Processo 0 lê todos os elementos e distribui com Scatter */
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

/*
 * RESUMO DA OTIMIZAÇÃO DE PONTEIROS:
 *
 * Versão original (com cópia):
 *   merge_result[] → copia para local_A[]   → O(local_n) operações
 *
 * Versão Q14 (com troca de ponteiros):
 *   Escreve em free_buf → troca ponteiros    → O(1) operações
 *
 * Para local_n=10^6, a diferença é: 10^6 cópias vs 2 atribuições de ponteiro.
 * O ganho é especialmente relevante em muitas fases e dados grandes.
 */
