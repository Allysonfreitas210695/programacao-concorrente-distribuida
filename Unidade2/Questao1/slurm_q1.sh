#!/bin/bash
#SBATCH --job-name=omp_trap_q1
#SBATCH --output=saida_q1.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --partition=sequana_cpu_dev

module load gcc

gcc -fopenmp -o omp_trap_1 omp_trap_1.c -lm

for threads in 1 2 4 8 16; do
  for n in 1000 10000 100000 1000000; do
    export OMP_NUM_THREADS=$threads
    echo "Threads=$threads, n=$n:"
    ./omp_trap_1 $n
  done
done
