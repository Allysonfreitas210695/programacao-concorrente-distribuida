/*
 * mpi_odd_even_timing.c
 * Ordenação ímpar-par paralela com medição de tempo via MPI_Wtime.
 * Não imprime o vetor — apenas o tempo de ordenação.
 *
 * Compile:  mpicc -O2 -o mpi_odd_even_timing mpi_odd_even_timing.c
 * Run:      mpiexec -n <p> ./mpi_odd_even_timing <global_n>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

const int RMAX = 100;

int  Compare(const void* a_p, const void* b_p);
void Generate_list(int local_A[], int local_n, int my_rank);
void Sort(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm);
void Odd_even_iter(int local_A[], int temp_B[], int temp_C[],
        int local_n, int phase, int even_partner, int odd_partner,
        int my_rank, int p, MPI_Comm comm);
void Merge_low(int my_keys[], int recv_keys[], int temp_keys[], int local_n);
void Merge_high(int local_A[], int temp_B[], int temp_C[], int local_n);


int main(int argc, char* argv[]) {
   int my_rank, p;
   int *local_A;
   int global_n, local_n;
   double t_start, t_elapsed, t_max;
   MPI_Comm comm = MPI_COMM_WORLD;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(comm, &p);
   MPI_Comm_rank(comm, &my_rank);

   if (my_rank == 0) {
      if (argc != 2) {
         fprintf(stderr, "Uso: mpirun -np <p> %s <global_n>\n", argv[0]);
         global_n = -1;
      } else {
         global_n = atoi(argv[1]);
         if (global_n % p != 0) {
            fprintf(stderr, "Erro: global_n (%d) deve ser divisivel por p (%d)\n",
                    global_n, p);
            global_n = -1;
         }
      }
   }

   MPI_Bcast(&global_n, 1, MPI_INT, 0, comm);
   if (global_n <= 0) { MPI_Finalize(); return 1; }

   local_n = global_n / p;
   local_A = (int*) malloc(local_n * sizeof(int));
   Generate_list(local_A, local_n, my_rank);

   MPI_Barrier(comm);
   t_start = MPI_Wtime();

   Sort(local_A, local_n, my_rank, p, comm);

   t_elapsed = MPI_Wtime() - t_start;
   MPI_Reduce(&t_elapsed, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

   if (my_rank == 0)
      printf("p=%d  n=%d  tempo=%.6f s\n", p, global_n, t_max);

   free(local_A);
   MPI_Finalize();
   return 0;
}


void Generate_list(int local_A[], int local_n, int my_rank) {
   srandom(my_rank + 1);
   for (int i = 0; i < local_n; i++)
      local_A[i] = random() % RMAX;
}

int Compare(const void* a_p, const void* b_p) {
   int a = *((int*)a_p), b = *((int*)b_p);
   return (a < b) ? -1 : (a > b) ? 1 : 0;
}

void Sort(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm) {
   int *temp_B = malloc(local_n * sizeof(int));
   int *temp_C = malloc(local_n * sizeof(int));
   int even_partner, odd_partner;

   if (my_rank % 2 != 0) {
      even_partner = my_rank - 1;
      odd_partner  = (my_rank + 1 == p) ? MPI_PROC_NULL : my_rank + 1;
   } else {
      even_partner = (my_rank + 1 == p) ? MPI_PROC_NULL : my_rank + 1;
      odd_partner  = my_rank - 1;
   }

   qsort(local_A, local_n, sizeof(int), Compare);

   for (int phase = 0; phase < p; phase++)
      Odd_even_iter(local_A, temp_B, temp_C, local_n, phase,
                    even_partner, odd_partner, my_rank, p, comm);

   free(temp_B);
   free(temp_C);
}

void Odd_even_iter(int local_A[], int temp_B[], int temp_C[],
        int local_n, int phase, int even_partner, int odd_partner,
        int my_rank, int p, MPI_Comm comm) {
   MPI_Status status;
   int partner = (phase % 2 == 0) ? even_partner : odd_partner;

   if (partner >= 0) {
      MPI_Sendrecv(local_A, local_n, MPI_INT, partner, 0,
                   temp_B,  local_n, MPI_INT, partner, 0, comm, &status);
      int keep_low = (phase % 2 == 0) ? (my_rank % 2 == 0) : (my_rank % 2 != 0);
      if (keep_low)
         Merge_low(local_A, temp_B, temp_C, local_n);
      else
         Merge_high(local_A, temp_B, temp_C, local_n);
   }
}

void Merge_low(int my_keys[], int recv_keys[], int temp_keys[], int local_n) {
   int m_i = 0, r_i = 0, t_i = 0;
   while (t_i < local_n) {
      if (my_keys[m_i] <= recv_keys[r_i])
         temp_keys[t_i++] = my_keys[m_i++];
      else
         temp_keys[t_i++] = recv_keys[r_i++];
   }
   memcpy(my_keys, temp_keys, local_n * sizeof(int));
}

void Merge_high(int local_A[], int temp_B[], int temp_C[], int local_n) {
   int ai = local_n-1, bi = local_n-1, ci = local_n-1;
   while (ci >= 0) {
      if (local_A[ai] >= temp_B[bi])
         temp_C[ci--] = local_A[ai--];
      else
         temp_C[ci--] = temp_B[bi--];
   }
   memcpy(local_A, temp_C, local_n * sizeof(int));
}
