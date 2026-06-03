/* File:    omp_trap_1.c
 * Purpose: Estimate definite integral (or area under curve) using trapezoidal
 *          rule.
 *
 * MODIFICADO: diretiva critical REMOVIDA propositalmente para demonstrar
 *             race condition (condição de corrida) — Questão 1 da lista.
 *
 * Compile: gcc -g -Wall -fopenmp -o omp_trap_1 omp_trap_1.c
 * Usage:   ./omp_trap_1 <number of threads>
 *
 * IPP:  Section 5.2.1 (pp. 216 and ff.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void Usage(char* prog_name);
double f(double x);
void Trap(double a, double b, int n, double* global_result_p);

int main(int argc, char* argv[]) {
   double  global_result = 0.0;
   double  a, b;
   int     n;
   int     thread_count;

   if (argc != 2) Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);
   printf("Enter a, b, and n\n");
   scanf("%lf %lf %d", &a, &b, &n);
   if (n % thread_count != 0) Usage(argv[0]);

#  pragma omp parallel num_threads(thread_count)
   Trap(a, b, n, &global_result);

   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n",
      a, b, global_result);
   return 0;
}

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
   fprintf(stderr, "   number of trapezoids must be evenly divisible by\n");
   fprintf(stderr, "   number of threads\n");
   exit(0);
}

double f(double x) {
   return x * x;
}

void Trap(double a, double b, int n, double* global_result_p) {
   double  h, x, my_result;
   double  local_a, local_b;
   int  i, local_n;
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();

   h = (b - a) / n;
   local_n = n / thread_count;
   local_a = a + my_rank * local_n * h;
   local_b = local_a + local_n * h;
   my_result = (f(local_a) + f(local_b)) / 2.0;
   for (i = 1; i <= local_n - 1; i++) {
      x = local_a + i * h;
      my_result += f(x);
   }
   my_result = my_result * h;

   /* CRITICAL REMOVIDA — race condition intencional para a questão 1 */
   *global_result_p += my_result;
}
