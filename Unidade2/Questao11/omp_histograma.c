/* File:      omp_histograma.c
 * Purpose:   Build a histogram from some random data using OpenMP
 *            Based on: ipp-source/ch2/histogram.c (IPP Section 2.7.1)
 *
 * Compile:   gcc -g -Wall -fopenmp -o omp_histograma omp_histograma.c
 * Run:       ./omp_histograma <bin_count> <min_meas> <max_meas> <data_count> <thread_count>
 *
 * Input:     None
 * Output:    A histogram with X's showing the number of measurements
 *            in each bin
 *
 * Notes:
 * 1.  Actual measurements y are in the range min_meas <= y < max_meas
 * 2.  bin_counts[i] stores the number of measurements x in the range
 * 3.  bin_maxes[i-1] <= x < bin_maxes[i] (bin_maxes[-1] = min_meas)
 * 4.  Parallelization: each thread maintains a local bin_counts array
 *     to avoid race conditions; local arrays are merged via critical section
 *
 * IPP:  Section 2.7.1 (pp. 66 and ff.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void Usage(char prog_name[]);

void Get_args(
      char*    argv[]        /* in  */,
      int*     bin_count_p   /* out */,
      float*   min_meas_p    /* out */,
      float*   max_meas_p    /* out */,
      int*     data_count_p  /* out */,
      int*     thread_count_p /* out */);

void Gen_data(
      float   min_meas    /* in  */,
      float   max_meas    /* in  */,
      float   data[]      /* out */,
      int     data_count  /* in  */);

void Gen_bins(
      float min_meas      /* in  */,
      float max_meas      /* in  */,
      float bin_maxes[]   /* out */,
      int   bin_counts[]  /* out */,
      int   bin_count     /* in  */);

int Which_bin(
      float    data         /* in */,
      float    bin_maxes[]  /* in */,
      int      bin_count    /* in */,
      float    min_meas     /* in */);

void Print_histo(
      float    bin_maxes[]   /* in */,
      int      bin_counts[]  /* in */,
      int      bin_count     /* in */,
      float    min_meas      /* in */);


int main(int argc, char* argv[]) {
   int bin_count, i, bin;
   float min_meas, max_meas;
   float* bin_maxes;
   int* bin_counts;
   int data_count;
   float* data;
   int thread_count;

   /* Check and get command line args */
   if (argc != 6) Usage(argv[0]);
   Get_args(argv, &bin_count, &min_meas, &max_meas, &data_count, &thread_count);

   /* Allocate arrays needed */
   bin_maxes  = malloc(bin_count * sizeof(float));
   bin_counts = malloc(bin_count * sizeof(int));
   data       = malloc(data_count * sizeof(float));

   /* Generate the data */
   Gen_data(min_meas, max_meas, data, data_count);

   /* Create bins for storing counts */
   Gen_bins(min_meas, max_meas, bin_maxes, bin_counts, bin_count);

   double inicio = omp_get_wtime();

   /* ---------------------------------------------------------------
    * REGIÃO PARALELA
    * Cada thread usa um bin_counts local para evitar race condition
    * no array global. Ao final, reduz para o global via critical.
    * --------------------------------------------------------------- */
   #pragma omp parallel num_threads(thread_count) \
       private(i, bin) shared(data, bin_maxes, bin_counts, bin_count, data_count, min_meas)
   {
      /* Histograma local desta thread, zerado com calloc */
      int* local_counts = (int*)calloc(bin_count, sizeof(int));

      /* Divide as iterações entre as threads */
      #pragma omp for
      for (i = 0; i < data_count; i++) {
         bin = Which_bin(data[i], bin_maxes, bin_count, min_meas);
         /* Incremento no array local — sem risco de corrida */
         local_counts[bin]++;
      }

      /* Reduz o histograma local para o global uma única vez por thread */
      #pragma omp critical
      {
         for (int b = 0; b < bin_count; b++)
            bin_counts[b] += local_counts[b];
      }

      free(local_counts);
   } /* Fim da região paralela */

   double fim = omp_get_wtime();

#  ifdef DEBUG
   printf("bin_counts = ");
   for (i = 0; i < bin_count; i++)
      printf("%d ", bin_counts[i]);
   printf("\n");
#  endif

   /* Print the histogram */
   Print_histo(bin_maxes, bin_counts, bin_count, min_meas);

   printf("\nThreads: %d | Tempo: %.4f s\n", thread_count, fim - inicio);

   free(data);
   free(bin_maxes);
   free(bin_counts);
   return 0;

}  /* main */


/*---------------------------------------------------------------------
 * Function:  Usage
 */
void Usage(char prog_name[] /* in */) {
   fprintf(stderr, "usage: %s ", prog_name);
   fprintf(stderr, "<bin_count> <min_meas> <max_meas> <data_count> <thread_count>\n");
   exit(0);
}  /* Usage */


/*---------------------------------------------------------------------
 * Function:  Get_args
 */
void Get_args(
      char*    argv[]         /* in  */,
      int*     bin_count_p    /* out */,
      float*   min_meas_p     /* out */,
      float*   max_meas_p     /* out */,
      int*     data_count_p   /* out */,
      int*     thread_count_p /* out */) {

   *bin_count_p    = strtol(argv[1], NULL, 10);
   *min_meas_p     = strtof(argv[2], NULL);
   *max_meas_p     = strtof(argv[3], NULL);
   *data_count_p   = strtol(argv[4], NULL, 10);
   *thread_count_p = strtol(argv[5], NULL, 10);

#  ifdef DEBUG
   printf("bin_count = %d\n", *bin_count_p);
   printf("min_meas = %f, max_meas = %f\n", *min_meas_p, *max_meas_p);
   printf("data_count = %d\n", *data_count_p);
   printf("thread_count = %d\n", *thread_count_p);
#  endif
}  /* Get_args */


/*---------------------------------------------------------------------
 * Function:  Gen_data  (inalterado do serial)
 */
void Gen_data(
        float   min_meas    /* in  */,
        float   max_meas    /* in  */,
        float   data[]      /* out */,
        int     data_count  /* in  */) {
   int i;

   srandom(0);
   for (i = 0; i < data_count; i++)
      data[i] = min_meas + (max_meas - min_meas)*random()/((double) RAND_MAX);

#  ifdef DEBUG
   printf("data = ");
   for (i = 0; i < data_count; i++)
      printf("%4.3f ", data[i]);
   printf("\n");
#  endif
}  /* Gen_data */


/*---------------------------------------------------------------------
 * Function:  Gen_bins  (inalterado do serial)
 */
void Gen_bins(
      float min_meas      /* in  */,
      float max_meas      /* in  */,
      float bin_maxes[]   /* out */,
      int   bin_counts[]  /* out */,
      int   bin_count     /* in  */) {
   float bin_width;
   int   i;

   bin_width = (max_meas - min_meas)/bin_count;

   for (i = 0; i < bin_count; i++) {
      bin_maxes[i]  = min_meas + (i+1)*bin_width;
      bin_counts[i] = 0;
   }

#  ifdef DEBUG
   printf("bin_maxes = ");
   for (i = 0; i < bin_count; i++)
      printf("%4.3f ", bin_maxes[i]);
   printf("\n");
#  endif
}  /* Gen_bins */


/*---------------------------------------------------------------------
 * Function:  Which_bin  (inalterado do serial)
 */
int Which_bin(
      float   data          /* in */,
      float   bin_maxes[]   /* in */,
      int     bin_count     /* in */,
      float   min_meas      /* in */) {
   int bottom = 0, top = bin_count-1;
   int mid;
   float bin_max, bin_min;

   while (bottom <= top) {
      mid     = (bottom + top)/2;
      bin_max = bin_maxes[mid];
      bin_min = (mid == 0) ? min_meas : bin_maxes[mid-1];
      if (data >= bin_max)
         bottom = mid+1;
      else if (data < bin_min)
         top = mid-1;
      else
         return mid;
   }

   fprintf(stderr, "Data = %f doesn't belong to a bin!\n", data);
   fprintf(stderr, "Quitting\n");
   exit(-1);
}  /* Which_bin */


/*---------------------------------------------------------------------
 * Function:  Print_histo  (inalterado do serial)
 */
void Print_histo(
        float  bin_maxes[]   /* in */,
        int    bin_counts[]  /* in */,
        int    bin_count     /* in */,
        float  min_meas      /* in */) {
   int i, j;
   float bin_max, bin_min;

   for (i = 0; i < bin_count; i++) {
      bin_max = bin_maxes[i];
      bin_min = (i == 0) ? min_meas : bin_maxes[i-1];
      printf("%.3f-%.3f:\t", bin_min, bin_max);
      for (j = 0; j < bin_counts[i]; j++)
         printf("X");
      printf("\n");
   }
}  /* Print_histo */
