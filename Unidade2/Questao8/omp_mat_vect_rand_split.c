/* File:     omp_mat_vect_rand_split.c
 *
 * Purpose:
 *   Calcula o produto paralelo de uma matriz m×n pelo vetor coluna x (n×1),
 *   gerando o vetor resultado y (m×1).
 *   A matriz é distribuída por blocos de linhas entre as threads.
 *   Os valores de A e x são gerados aleatoriamente (sem entrada do usuário).
 *
 * Compile:
 *   gcc -g -O2 -Wall -fopenmp -o omp_mat_vect_rand_split omp_mat_vect_rand_split.c
 *   (Com Valgrind/cachegrind para perfilamento de cache — Questão 8)
 *
 * Usage:
 *   ./omp_mat_vect_rand_split <thread_count> <m> <n>
 *
 * IPP: Exercise 5.12
 */

/* Inclui funções de entrada/saída padrão */
#include <stdio.h>
/* Inclui malloc, free, strtol, exit, random */
#include <stdlib.h>
/* Inclui a API OpenMP */
#include <omp.h>
/* Inclui macros GET_TIME para cronometrar com precisão usando clock_gettime */
#include "timer.h"

/* ---- Declarações das funções (protótipos) ---- */

/* Lê os argumentos da linha de comando */
void Get_args(int argc, char* argv[], int* thread_count_p,
      int* m_p, int* n_p);

/* Exibe mensagem de uso correto e termina o programa */
void Usage(char* prog_name);

/* Preenche a matriz A com valores aleatórios */
void Gen_matrix(double A[], int m, int n);

/* Lê a matriz A do teclado (usada apenas com flag DEBUG) */
void Read_matrix(char* prompt, double A[], int m, int n);

/* Preenche o vetor x com valores aleatórios */
void Gen_vector(double x[], int n);

/* Lê o vetor x do teclado (usada apenas com flag DEBUG) */
void Read_vector(char* prompt, double x[], int n);

/* Imprime a matriz A no terminal */
void Print_matrix(char* title, double A[], int m, int n);

/* Imprime o vetor y no terminal */
void Print_vector(char* title, double y[], double m);

/* Realiza o produto matriz-vetor em paralelo com OpenMP */
void Omp_mat_vect(double A[], double x[], double y[],
      int m, int n, int thread_count);

/* ------------------------------------------------------------------ */
int main(int argc, char* argv[]) {
   int     thread_count; /* Número de threads OpenMP */
   int     m;            /* Número de linhas da matriz A */
   int     n;            /* Número de colunas da matriz A (e tamanho de x) */
   double* A;            /* Ponteiro para a matriz A (armazenada linearmente) */
   double* x;            /* Ponteiro para o vetor de entrada x (tamanho n) */
   double* y;            /* Ponteiro para o vetor resultado y (tamanho m) */

   /* Processa os argumentos da linha de comando e preenche thread_count, m, n */
   Get_args(argc, argv, &thread_count, &m, &n);

   /* Aloca a matriz A como array 1D de m*n doubles.
    * O elemento A[i][j] é acessado como A[i*n + j] */
   A = malloc(m * n * sizeof(double));

   /* Aloca o vetor x com n doubles */
   x = malloc(n * sizeof(double));

   /* Aloca o vetor resultado y com m doubles */
   y = malloc(m * sizeof(double));

   /* Se compilado com -DDEBUG, lê A e x do teclado e exibe o que foi lido */
#  ifdef DEBUG
   Read_matrix("Enter the matrix", A, m, n);
   Print_matrix("We read", A, m, n);
   Read_vector("Enter the vector", x, n);
   Print_vector("We read", x, n);
#  else
   /* Modo normal: gera A e x com valores aleatórios */
   Gen_matrix(A, m, n);  /* Preenche todos os m*n elementos de A */
   Gen_vector(x, n);     /* Preenche os n elementos de x */
#  endif

   /* Executa o produto matriz-vetor paralelo: y = A × x */
   Omp_mat_vect(A, x, y, m, n, thread_count);

   /* Se DEBUG, exibe o vetor resultado y */
#  ifdef DEBUG
   Print_vector("The product is", y, m);
#  endif

   /* Libera a memória alocada para A, x e y */
   free(A);
   free(x);
   free(y);

   return 0; /* Programa encerrado com sucesso */
}  /* main */


/* ------------------------------------------------------------------
 * Função:  Get_args
 * Propósito: Lê e valida os argumentos da linha de comando.
 *            Espera exatamente 3 argumentos: thread_count, m, n.
 */
void Get_args(int argc, char* argv[], int* thread_count_p,
      int* m_p, int* n_p) {

   /* Se não vieram exatamente 3 argumentos, exibe uso e encerra */
   if (argc != 4) Usage(argv[0]);

   /* Converte argv[1] para inteiro: número de threads */
   *thread_count_p = strtol(argv[1], NULL, 10);

   /* Converte argv[2] para inteiro: número de linhas da matriz (m) */
   *m_p = strtol(argv[2], NULL, 10);

   /* Converte argv[3] para inteiro: número de colunas da matriz (n) */
   *n_p = strtol(argv[3], NULL, 10);

   /* Valida que todos os valores são positivos */
   if (*thread_count_p <= 0 || *m_p <= 0 || *n_p <= 0) Usage(argv[0]);

}  /* Get_args */

/* ------------------------------------------------------------------
 * Função:  Usage
 * Propósito: Exibe mensagem de uso correto no stderr e encerra.
 */
void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count> <m> <n>\n", prog_name);
   exit(0); /* Encerra o programa imediatamente */
}  /* Usage */

/* ------------------------------------------------------------------
 * Função:    Read_matrix
 * Propósito: Lê a matriz A do teclado (linha a linha, coluna a coluna).
 *            Usada apenas quando compilado com -DDEBUG.
 */
void Read_matrix(char* prompt, double A[], int m, int n) {
   int i, j;

   printf("%s\n", prompt); /* Exibe a mensagem de prompt para o usuário */
   for (i = 0; i < m; i++)         /* Para cada linha i */
      for (j = 0; j < n; j++)      /* Para cada coluna j */
         scanf("%lf", &A[i*n+j]);  /* Lê e armazena em A[i][j] = A[i*n+j] */
}  /* Read_matrix */

/* ------------------------------------------------------------------
 * Função: Gen_matrix
 * Propósito: Preenche a matriz A com valores pseudoaleatórios em [0, 1).
 *            Usa random() que gera inteiros; divide por RAND_MAX para [0,1).
 */
void Gen_matrix(double A[], int m, int n) {
   int i, j;
   for (i = 0; i < m; i++)              /* Para cada linha */
      for (j = 0; j < n; j++)           /* Para cada coluna */
         A[i*n+j] = random() / ((double) RAND_MAX); /* Valor em [0.0, 1.0) */
}  /* Gen_matrix */

/* ------------------------------------------------------------------
 * Função: Gen_vector
 * Propósito: Preenche o vetor x com valores pseudoaleatórios em [0, 1).
 */
void Gen_vector(double x[], int n) {
   int i;
   for (i = 0; i < n; i++)
      x[i] = random() / ((double) RAND_MAX); /* Valor em [0.0, 1.0) */
}  /* Gen_vector */

/* ------------------------------------------------------------------
 * Função:        Read_vector
 * Propósito:     Lê o vetor x do teclado elemento a elemento.
 *                Usada apenas com -DDEBUG.
 */
void Read_vector(char* prompt, double x[], int n) {
   int i;

   printf("%s\n", prompt); /* Exibe prompt para o usuário */
   for (i = 0; i < n; i++)
      scanf("%lf", &x[i]); /* Lê o i-ésimo elemento de x */
}  /* Read_vector */


/* ------------------------------------------------------------------
 * Função:  Omp_mat_vect
 * Propósito: Calcula y = A × x em paralelo com OpenMP.
 *            A matriz A (m×n) é distribuída por blocos de linhas:
 *            cada thread processa um conjunto contíguo de linhas de A
 *            e calcula as posições correspondentes de y.
 *
 * Parâmetros:
 *   A            - matriz m×n armazenada como array 1D (A[i][j] = A[i*n+j])
 *   x            - vetor de entrada (tamanho n), compartilhado (somente leitura)
 *   y            - vetor resultado (tamanho m), cada y[i] escrito por 1 thread
 *   m            - número de linhas de A (e tamanho de y)
 *   n            - número de colunas de A (e tamanho de x)
 *   thread_count - número de threads OpenMP
 */
void Omp_mat_vect(double A[], double x[], double y[],
      int m, int n, int thread_count) {
   int i, j;
   double start, finish, elapsed, temp;

   /* Marca o início do cronômetro usando clock_gettime (via macro em timer.h) */
   GET_TIME(start);

   /* REGIÃO PARALELA:
    * - default(none): força declarar explicitamente o escopo de cada variável
    * - private(i, j, temp): cada thread tem sua própria cópia dessas variáveis
    * - shared(A, x, y, m, n): todas as threads compartilham esses dados
    *
    * O laço externo (for i) é dividido em blocos de linhas entre as threads.
    * Thread t processa as linhas [t*chunk .. (t+1)*chunk - 1].
    * Como y[i] é escrito por apenas uma thread, não há race condition em y. */
   #pragma omp parallel for num_threads(thread_count)  \
       default(none) private(i, j, temp)  shared(A, x, y, m, n)
   for (i = 0; i < m; i++) {
      y[i] = 0.0; /* Inicializa y[i] antes de acumular o produto da linha i */

      /* Produto escalar da linha i de A com o vetor x:
       * y[i] = A[i][0]*x[0] + A[i][1]*x[1] + ... + A[i][n-1]*x[n-1] */
      for (j = 0; j < n; j++) {
         temp = A[i*n+j] * x[j]; /* Multiplica o elemento A[i][j] por x[j] */
         y[i] += temp;           /* Acumula no resultado da linha i */
      }
   }
   /* Barreira implícita: todas as threads concluem antes de sair do parallel for */

   /* Marca o fim do cronômetro */
   GET_TIME(finish);

   /* Calcula o tempo decorrido em segundos */
   elapsed = finish - start;

   /* Exibe o tempo de execução do produto matriz-vetor paralelo */
   printf("Elapsed time = %e seconds\n", elapsed);

}  /* Omp_mat_vect */


/* ------------------------------------------------------------------
 * Função:    Print_matrix
 * Propósito: Imprime a matriz A (m×n) formatada linha a linha.
 *            Usada apenas para depuração com -DDEBUG.
 */
void Print_matrix(char* title, double A[], int m, int n) {
   int i, j;

   printf("%s\n", title); /* Exibe o título/rótulo da matriz */
   for (i = 0; i < m; i++) {          /* Para cada linha */
      for (j = 0; j < n; j++)         /* Para cada coluna */
         printf("%4.1f ", A[i*n + j]); /* Imprime A[i][j] com 1 casa decimal */
      printf("\n"); /* Quebra de linha após cada linha da matriz */
   }
}  /* Print_matrix */


/* ------------------------------------------------------------------
 * Função:    Print_vector
 * Propósito: Imprime o vetor y elemento a elemento em uma linha.
 *            Usada apenas para depuração com -DDEBUG.
 */
void Print_vector(char* title, double y[], double m) {
   int i;

   printf("%s\n", title); /* Exibe o título/rótulo do vetor */
   for (i = 0; i < m; i++)
      printf("%4.1f ", y[i]); /* Imprime cada elemento com 1 casa decimal */
   printf("\n"); /* Quebra de linha ao final */
}  /* Print_vector */
