/* File:    omp_trap_schedule.c
 * Purpose: Modificação de omp_trap3.c para usar schedule(runtime) e
 *          registrar qual thread executa cada iteração — Questão 6.
 *
 * Compile: gcc -g -Wall -fopenmp -o omp_trap_schedule omp_trap_schedule.c -lm
 * Usage:   ./omp_trap_schedule <number of threads>
 *          OMP_SCHEDULE="static"    ./omp_trap_schedule 4
 *          OMP_SCHEDULE="static,2"  ./omp_trap_schedule 4
 *          OMP_SCHEDULE="dynamic"   ./omp_trap_schedule 4
 *          OMP_SCHEDULE="guided"    ./omp_trap_schedule 4
 *
 * IPP:  Section 5.5 (pp. 224 and ff.) — modificado para Questão 6
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void Usage(char* prog_name);
double f(double x);
double Trap(double a, double b, int n, int thread_count, int* iteracoes);

int main(int argc, char* argv[]) {
   double  global_result = 0.0;
   double  a, b;
   int     n;
   int     thread_count;

   if (argc != 2) Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);
   printf("Enter a, b, and n\n");
   scanf("%lf %lf %d", &a, &b, &n);

   int* iteracoes = (int*)malloc(n * sizeof(int));

   global_result = Trap(a, b, n, thread_count, iteracoes);

   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n", a, b, global_result);

   /* Mostra quais iterações cada thread executou */
   printf("\nDistribuicao de iteracoes:\n");
   for (int t = 0; t < thread_count; t++) {
      printf("Thread %d: ", t);
      for (int i = 1; i < n; i++)
         if (iteracoes[i] == t)
            printf("%d ", i);
      printf("\n");
   }

   free(iteracoes);
   return 0;
}

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
   exit(0);
}

double f(double x) {
   return x * x;
}

double Trap(double a, double b, int n, int thread_count, int* iteracoes) {
   double  h, approx;
   int  i;

   h = (b - a) / n;
   approx = (f(a) + f(b)) / 2.0;

#  pragma omp parallel for num_threads(thread_count) \
         schedule(runtime) reduction(+: approx)
   for (i = 1; i <= n - 1; i++) {
      iteracoes[i] = omp_get_thread_num();
      approx += f(a + i * h);
   }

   approx = h * approx;
   return approx;
}
