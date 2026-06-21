/*
 * Questão 13 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Arquivo original: mpi_odd_even.c (livro IPP - Pacheco, cap. 3)
 *
 * Objetivo: Implementar o algoritmo de ordenação paralela
 * "Odd-Even Transposition Sort" (Ordenação por Transposição Ímpar-Par).
 *
 * Ideia do algoritmo:
 *   Cada processo mantém uma fatia local de local_n elementos já ordenados.
 *   Em p fases alternadas (par e ímpar), processos vizinhos trocam seus arrays
 *   e executam um merge-split: cada um fica com os menores (Merge_low) ou
 *   maiores (Merge_high) elementos do resultado do merge.
 *   Após p fases, a lista global está ordenada.
 *
 * Prova informal: em p fases, qualquer elemento pode "viajar" no máximo p
 * posições de processo — o suficiente para chegar ao processo correto.
 *
 * Compile: mpicc -g -Wall -o mpi_odd_even mpi_odd_even.c
 * Run:     mpiexec -n <p> ./mpi_odd_even <g|i> <global_n>
 *            g = gerar aleatoriamente, i = ler do stdin
 *            global_n deve ser divisível por p
 */

#include <stdio.h>    /* printf, fprintf, scanf */
#include <stdlib.h>   /* malloc, free, atoi, qsort */
#include <string.h>   /* memcpy */
#include <mpi.h>

/* Valor máximo dos inteiros gerados aleatoriamente */
const int RMAX = 100;

/* Protótipos das funções locais (sem comunicação MPI) */
void Usage(char* program);
void Print_list(int local_A[], int local_n, int rank);
void Merge_low(int local_A[], int temp_B[], int temp_C[], int local_n);
void Merge_high(int local_A[], int temp_B[], int temp_C[], int local_n);
void Generate_list(int local_A[], int local_n, int my_rank);
int  Compare(const void* a_p, const void* b_p);

/* Protótipos das funções que envolvem comunicação MPI */
void Get_args(int argc, char* argv[], int* global_n_p, int* local_n_p,
         char* gi_p, int my_rank, int p, MPI_Comm comm);
void Sort(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm);
void Odd_even_iter(int local_A[], int temp_B[], int temp_C[],
         int local_n, int phase, int even_partner, int odd_partner,
         int my_rank, int p, MPI_Comm comm);
void Print_local_lists(int local_A[], int local_n,
         int my_rank, int p, MPI_Comm comm);
void Print_global_list(int local_A[], int local_n, int my_rank,
         int p, MPI_Comm comm);
void Read_list(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm);


/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    int my_rank, p;   /* rank deste processo e total de processos */
    char g_i;         /* 'g' = gerar aleatório, 'i' = ler do usuário */
    int *local_A;     /* array local com local_n elementos */
    int global_n;     /* tamanho total do array global */
    int local_n;      /* tamanho do array local = global_n / p */
    MPI_Comm comm;    /* comunicador (atalho para MPI_COMM_WORLD) */

    /* Inicializa o ambiente MPI */
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);        /* total de processos */
    MPI_Comm_rank(comm, &my_rank);  /* rank deste processo */

    /* Lê e valida os argumentos da linha de comando */
    Get_args(argc, argv, &global_n, &local_n, &g_i, my_rank, p, comm);

    /* Aloca o array local */
    local_A = (int*) malloc(local_n * sizeof(int));

    if (g_i == 'g') {
        /* 'g': gera lista aleatória e a imprime (antes de ordenar) */
        Generate_list(local_A, local_n, my_rank);
        Print_local_lists(local_A, local_n, my_rank, p, comm);
    } else {
        /* 'i': processo 0 lê do stdin e distribui com Scatter */
        Read_list(local_A, local_n, my_rank, p, comm);
#ifdef DEBUG
        /* Modo debug: imprime as listas locais antes de ordenar */
        Print_local_lists(local_A, local_n, my_rank, p, comm);
#endif
    }

#ifdef DEBUG
    printf("Proc %d > Before Sort\n", my_rank);
    fflush(stdout);
#endif

    /* Ordena usando o algoritmo odd-even transposition sort paralelo */
    Sort(local_A, local_n, my_rank, p, comm);

#ifdef DEBUG
    Print_local_lists(local_A, local_n, my_rank, p, comm);
    fflush(stdout);
#endif

    /* Reúne e imprime a lista global ordenada (somente processo 0) */
    Print_global_list(local_A, local_n, my_rank, p, comm);

    free(local_A);
    MPI_Finalize();
    return 0;
}


/*-------------------------------------------------------------------
 * Generate_list
 *
 * Preenche local_A[] com local_n inteiros aleatórios entre 0 e RMAX-1.
 * A semente (my_rank+1) garante que cada processo gera valores diferentes,
 * mantendo reprodutibilidade por processo.
 */
void Generate_list(int local_A[], int local_n, int my_rank) {
    int i;
    srandom(my_rank + 1);   /* semente diferente por processo */
    for (i = 0; i < local_n; i++)
        local_A[i] = random() % RMAX;
}


/*-------------------------------------------------------------------
 * Usage
 *
 * Imprime a sintaxe correta de invocação no stderr.
 * Chamada apenas pelo processo 0 quando os argumentos são inválidos.
 */
void Usage(char* program) {
    fprintf(stderr, "usage:  mpirun -np <p> %s <g|i> <global_n>\n", program);
    fprintf(stderr, "   - p: the number of processes \n");
    fprintf(stderr, "   - g: generate random, distributed list\n");
    fprintf(stderr, "   - i: user will input list on process 0\n");
    fprintf(stderr, "   - global_n: number of elements in global list");
    fprintf(stderr, " (must be evenly divisible by p)\n");
    fflush(stderr);
}


/*-------------------------------------------------------------------
 * Get_args
 *
 * Processo 0 valida os argumentos da linha de comando.
 * Em seguida, distribui gi_p e global_n_p para todos via MPI_Bcast.
 * Se os argumentos forem inválidos, global_n é setado para -1,
 * o que faz todos os processos terminarem com exit(-1).
 */
void Get_args(int argc, char* argv[], int* global_n_p, int* local_n_p,
         char* gi_p, int my_rank, int p, MPI_Comm comm) {

    if (my_rank == 0) {
        if (argc != 3) {
            /* Número errado de argumentos */
            Usage(argv[0]);
            *global_n_p = -1;   /* sinaliza erro */
        } else {
            *gi_p = argv[1][0];   /* primeiro argumento: 'g' ou 'i' */
            if (*gi_p != 'g' && *gi_p != 'i') {
                Usage(argv[0]);
                *global_n_p = -1;
            } else {
                *global_n_p = atoi(argv[2]);   /* segundo argumento: tamanho global */
                if (*global_n_p % p != 0) {
                    /* global_n deve ser divisível por p para fatias iguais */
                    Usage(argv[0]);
                    *global_n_p = -1;
                }
            }
        }
    }

    /* Distribui o caractere 'g'/'i' para todos os processos */
    MPI_Bcast(gi_p, 1, MPI_CHAR, 0, comm);

    /* Distribui o tamanho global (ou -1 em caso de erro) */
    MPI_Bcast(global_n_p, 1, MPI_INT, 0, comm);

    /* Todos os processos verificam se houve erro */
    if (*global_n_p <= 0) {
        MPI_Finalize();
        exit(-1);
    }

    /* Cada processo calcula seu tamanho local */
    *local_n_p = *global_n_p / p;

#ifdef DEBUG
    printf("Proc %d > gi = %c, global_n = %d, local_n = %d\n",
       my_rank, *gi_p, *global_n_p, *local_n_p);
    fflush(stdout);
#endif
}


/*-------------------------------------------------------------------
 * Read_list
 *
 * Processo 0 lê global_n inteiros do stdin e os distribui com MPI_Scatter.
 * Os demais processos apenas recebem sua fatia de local_n elementos.
 */
void Read_list(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm) {
    int i;
    int *temp;

    if (my_rank == 0) {
        /* Aloca buffer temporário para a lista global inteira */
        temp = (int*) malloc(p * local_n * sizeof(int));
        printf("Enter the elements of the list\n");
        for (i = 0; i < p * local_n; i++)
            scanf("%d", &temp[i]);
    }

    /*
     * MPI_Scatter: processo 0 divide 'temp' em blocos de local_n ints
     * e envia um bloco para cada processo (incluindo ele mesmo).
     * O argumento 'temp' é ignorado nos processos não-raiz.
     */
    MPI_Scatter(temp, local_n, MPI_INT, local_A, local_n, MPI_INT, 0, comm);

    if (my_rank == 0) free(temp);
}


/*-------------------------------------------------------------------
 * Print_global_list
 *
 * Processo 0 usa MPI_Gather para reunir todos os arrays locais
 * e imprime a lista global completa.
 * Os demais processos apenas participam do Gather (chamada coletiva).
 *
 * ATENÇÃO: o buffer 'A' nos processos não-raiz não é alocado —
 * isso é válido pois MPI_Gather ignora o recv_buffer nos não-raiz.
 */
void Print_global_list(int local_A[], int local_n, int my_rank, int p,
      MPI_Comm comm) {
    int* A;
    int i, n;

    if (my_rank == 0) {
        n = p * local_n;
        A = (int*) malloc(n * sizeof(int));

        /* Reúne todas as fatias locais em A[] */
        MPI_Gather(local_A, local_n, MPI_INT, A, local_n, MPI_INT, 0, comm);

        printf("Global list:\n");
        for (i = 0; i < n; i++) printf("%d ", A[i]);
        printf("\n\n");
        free(A);
    } else {
        /* Processos não-raiz apenas participam do Gather */
        MPI_Gather(local_A, local_n, MPI_INT, A, local_n, MPI_INT, 0, comm);
    }
}


/*-------------------------------------------------------------------
 * Compare
 *
 * Comparador para qsort: retorna negativo, 0 ou positivo conforme
 * *a_p < *b_p, *a_p == *b_p, ou *a_p > *b_p.
 * Necessário pois qsort exige função de comparação genérica (void*).
 */
int Compare(const void* a_p, const void* b_p) {
    int a = *((int*)a_p);
    int b = *((int*)b_p);
    if (a < b)       return -1;
    else if (a == b) return  0;
    else             return  1;
}


/*-------------------------------------------------------------------
 * Sort
 *
 * Orquestra o algoritmo Odd-Even Transposition Sort:
 *   1. Ordena o array local com qsort (O(n log n) local).
 *   2. Executa p fases de odd-even, alternando parceiros pares/ímpares.
 *
 * Cálculo dos parceiros:
 *   Processo de rank PAR:
 *     - Fase par:  parceiro = my_rank + 1  (à direita)
 *     - Fase ímpar: parceiro = my_rank - 1  (à esquerda)
 *   Processo de rank ÍMPAR:
 *     - Fase par:  parceiro = my_rank - 1  (à esquerda)
 *     - Fase ímpar: parceiro = my_rank + 1  (à direita)
 *
 *   MPI_PROC_NULL é usado quando o parceiro estaria fora dos limites
 *   (rank < 0 ou rank >= p). MPI ignora operações com MPI_PROC_NULL.
 */
void Sort(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm) {
    int phase;
    int *temp_B, *temp_C;
    int even_partner;   /* parceiro nas fases pares */
    int odd_partner;    /* parceiro nas fases ímpares */

    /* Buffers temporários para o merge-split */
    temp_B = (int*) malloc(local_n * sizeof(int));   /* array recebido do parceiro */
    temp_C = (int*) malloc(local_n * sizeof(int));   /* scratch para o merge */

    /* Calcula parceiros com base na paridade do rank */
    if (my_rank % 2 != 0) {
        /* Rank ímpar: nas fases pares olha para a esquerda, nas ímpares para direita */
        even_partner = my_rank - 1;
        odd_partner  = my_rank + 1;
        if (odd_partner == p) odd_partner = MPI_PROC_NULL;   /* último processo: sem parceiro ímpar */
    } else {
        /* Rank par: nas fases pares olha para direita, nas ímpares para esquerda */
        even_partner = my_rank + 1;
        if (even_partner == p) even_partner = MPI_PROC_NULL; /* último processo par: sem parceiro */
        odd_partner  = my_rank - 1;
        /* my_rank == 0: odd_partner = -1, que é < 0, não é usado com >= 0 */
    }

    /* Passo 1: ordena o array local com qsort */
    qsort(local_A, local_n, sizeof(int), Compare);

#ifdef DEBUG
    printf("Proc %d > before loop in sort\n", my_rank);
    fflush(stdout);
#endif

    /* Passo 2: p fases de odd-even */
    for (phase = 0; phase < p; phase++)
        Odd_even_iter(local_A, temp_B, temp_C, local_n, phase,
                      even_partner, odd_partner, my_rank, p, comm);

    free(temp_B);
    free(temp_C);
}


/*-------------------------------------------------------------------
 * Odd_even_iter
 *
 * Uma fase do algoritmo odd-even transposition sort.
 *
 * Fase par  (phase % 2 == 0): usa even_partner.
 * Fase ímpar (phase % 2 != 0): usa odd_partner.
 *
 * Se o parceiro é MPI_PROC_NULL, o processo fica ocioso nesta fase
 * (MPI_Sendrecv com MPI_PROC_NULL é um no-op).
 *
 * Regra de quem fica com o quê:
 *   Processo de rank MENOR entre o par → Merge_low  (fica com os menores)
 *   Processo de rank MAIOR entre o par → Merge_high (fica com os maiores)
 *
 * Isso mantém a invariante de que o processo de rank menor sempre
 * tem elementos menores ou iguais aos do processo de rank maior.
 */
void Odd_even_iter(int local_A[], int temp_B[], int temp_C[],
        int local_n, int phase, int even_partner, int odd_partner,
        int my_rank, int p, MPI_Comm comm) {
    MPI_Status status;

    if (phase % 2 == 0) {
        /* ── Fase PAR ── */
        if (even_partner >= 0) {
            /*
             * MPI_Sendrecv: troca simultânea de arrays com even_partner.
             *   Envia local_A para even_partner.
             *   Recebe temp_B de even_partner.
             * Operação atômica — evita deadlock que ocorreria com Send+Recv.
             */
            MPI_Sendrecv(local_A, local_n, MPI_INT, even_partner, 0,
                         temp_B,  local_n, MPI_INT, even_partner, 0,
                         comm, &status);

            if (my_rank % 2 != 0)
                /* Rank ímpar: olhou para a esquerda (rank menor) → fica com os MAIORES */
                Merge_high(local_A, temp_B, temp_C, local_n);
            else
                /* Rank par: olhou para a direita (rank menor) → fica com os MENORES */
                Merge_low(local_A, temp_B, temp_C, local_n);
        }
    } else {
        /* ── Fase ÍMPAR ── */
        if (odd_partner >= 0) {
            MPI_Sendrecv(local_A, local_n, MPI_INT, odd_partner, 0,
                         temp_B,  local_n, MPI_INT, odd_partner, 0,
                         comm, &status);

            if (my_rank % 2 != 0)
                /* Rank ímpar: olhou para a direita (rank maior) → fica com os MENORES */
                Merge_low(local_A, temp_B, temp_C, local_n);
            else
                /* Rank par: olhou para a esquerda (rank maior) → fica com os MAIORES */
                Merge_high(local_A, temp_B, temp_C, local_n);
        }
    }
}


/*-------------------------------------------------------------------
 * Merge_low
 *
 * Faz o merge de local_A[] (meus elementos) com temp_B[] (elementos do parceiro),
 * mantendo os local_n MENORES em local_A[].
 *
 * Algoritmo: merge clássico de dois arrays ordenados, mas para quando
 * temp_keys já tem local_n elementos (não precisa terminar o merge completo).
 *
 * Após o merge, copia temp_keys de volta para local_A com memcpy.
 * (A Questão 14 elimina essa cópia usando troca de ponteiros.)
 *
 * Parâmetros:
 *   my_keys   [in/out]: array local (sobrescrito com os menores)
 *   recv_keys [in]:     array recebido do parceiro
 *   temp_keys [scratch]: buffer temporário para o resultado do merge
 *   local_n:  tamanho de cada array
 */
void Merge_low(int my_keys[], int recv_keys[], int temp_keys[], int local_n) {
    int m_i, r_i, t_i;

    /* Inicializa índices para percorrer os três arrays */
    m_i = r_i = t_i = 0;

    /* Merge: a cada passo escolhe o menor entre my_keys[m_i] e recv_keys[r_i] */
    while (t_i < local_n) {
        if (my_keys[m_i] <= recv_keys[r_i]) {
            /* Elemento de my_keys é menor ou igual — vai para temp */
            temp_keys[t_i] = my_keys[m_i];
            t_i++; m_i++;
        } else {
            /* Elemento de recv_keys é menor — vai para temp */
            temp_keys[t_i] = recv_keys[r_i];
            t_i++; r_i++;
        }
    }
    /* temp_keys[0..local_n-1] contém os local_n menores elementos */

    /* Copia o resultado de volta para my_keys */
    memcpy(my_keys, temp_keys, local_n * sizeof(int));
}


/*-------------------------------------------------------------------
 * Merge_high
 *
 * Faz o merge de local_A[] com temp_B[], mantendo os local_n MAIORES
 * em local_A[].
 *
 * Percorre os arrays de trás para frente para escolher os maiores primeiro.
 * Após o merge, copia temp_C de volta para local_A com memcpy.
 *
 * Parâmetros:
 *   local_A [in/out]: array local (sobrescrito com os maiores)
 *   temp_B  [in]:     array recebido do parceiro
 *   temp_C  [scratch]: buffer temporário para o resultado
 *   local_n: tamanho de cada array
 */
void Merge_high(int local_A[], int temp_B[], int temp_C[], int local_n) {
    int ai, bi, ci;

    /* Começa do final de cada array (elementos maiores) */
    ai = local_n - 1;
    bi = local_n - 1;
    ci = local_n - 1;

    /* Merge reverso: a cada passo escolhe o MAIOR entre local_A[ai] e temp_B[bi] */
    while (ci >= 0) {
        if (local_A[ai] >= temp_B[bi]) {
            /* Elemento de local_A é maior ou igual — vai para temp_C */
            temp_C[ci] = local_A[ai];
            ci--; ai--;
        } else {
            /* Elemento de temp_B é maior — vai para temp_C */
            temp_C[ci] = temp_B[bi];
            ci--; bi--;
        }
    }
    /* temp_C[0..local_n-1] contém os local_n maiores elementos em ordem crescente */

    /* Copia o resultado de volta para local_A */
    memcpy(local_A, temp_C, local_n * sizeof(int));
}


/*-------------------------------------------------------------------
 * Print_list
 *
 * Imprime os local_n elementos de local_A precedidos pelo rank.
 * Chamada apenas pelo processo 0 dentro de Print_local_lists.
 */
void Print_list(int local_A[], int local_n, int rank) {
    int i;
    printf("%d: ", rank);
    for (i = 0; i < local_n; i++) printf("%d ", local_A[i]);
    printf("\n");
}


/*-------------------------------------------------------------------
 * Print_local_lists
 *
 * Processo 0 coleta e imprime os arrays locais de todos os processos,
 * um por linha, na ordem dos ranks.
 *
 * Implementação via token:
 *   Processo 0: imprime seu próprio array, depois faz Recv de cada
 *               processo q=1..p-1 e imprime.
 *   Processos 1..p-1: fazem Send do seu array para o processo 0.
 *
 * Isso garante ordem de impressão, mas é sequencial (gargalo de escalabilidade).
 * Usada apenas para debug — não está no caminho crítico de desempenho.
 */
void Print_local_lists(int local_A[], int local_n,
         int my_rank, int p, MPI_Comm comm) {
    int*       A;
    int        q;
    MPI_Status status;

    if (my_rank == 0) {
        /* Aloca buffer para receber o array de cada processo */
        A = (int*) malloc(local_n * sizeof(int));

        /* Imprime o array do próprio processo 0 */
        Print_list(local_A, local_n, my_rank);

        /* Recebe e imprime os arrays dos demais processos em ordem */
        for (q = 1; q < p; q++) {
            MPI_Recv(A, local_n, MPI_INT, q, 0, comm, &status);
            Print_list(A, local_n, q);
        }
        free(A);
    } else {
        /* Cada processo não-raiz envia seu array para o processo 0 */
        MPI_Send(local_A, local_n, MPI_INT, 0, 0, comm);
    }
}

/*
 * EXEMPLO DO ALGORITMO com p=4 processos e local_n=2:
 *
 * Estado inicial (após qsort local):
 *   Proc 0: [3, 7]   Proc 1: [1, 9]   Proc 2: [2, 8]   Proc 3: [4, 6]
 *
 * Fase 0 (par): pares (0,1) e (2,3) trocam e fazem merge-split
 *   0↔1: [3,7]+[1,9] → Proc0 fica [1,3] (Merge_low), Proc1 fica [7,9] (Merge_high)
 *   2↔3: [2,8]+[4,6] → Proc2 fica [2,4] (Merge_low), Proc3 fica [6,8] (Merge_high)
 *
 * Fase 1 (ímpar): pares (1,2) trocam; procs 0 e 3 ficam ociosos
 *   1↔2: [7,9]+[2,4] → Proc1 fica [2,4] (Merge_low), Proc2 fica [7,9] (Merge_high)
 *
 * Fase 2 (par): pares (0,1) e (2,3) trocam
 *   0↔1: [1,3]+[2,4] → Proc0=[1,2], Proc1=[3,4]
 *   2↔3: [7,9]+[6,8] → Proc2=[6,7], Proc3=[8,9]
 *
 * Fase 3 (ímpar): pares (1,2) trocam
 *   1↔2: [3,4]+[6,7] → Proc1=[3,4], Proc2=[6,7]  (já ordenado, sem mudança)
 *
 * Resultado final: [1,2] [3,4] [6,7] [8,9] → lista global: 1 2 3 4 6 7 8 9 ✓
 */
