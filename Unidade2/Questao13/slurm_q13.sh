#!/bin/bash
#SBATCH --job-name=omp_count_sort_q13
#SBATCH --output=saida_q13.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --partition=sequana_cpu_dev

module load gcc
gcc -fopenmp -O2 -o omp_count_sort omp_count_sort.c

echo "=== n=5000 ==="
for t in 1 2 4 8 16; do
  echo -n "threads=$t: "
  ./omp_count_sort 5000 $t
done

echo "=== n=10000 ==="
for t in 1 2 4 8 16; do
  echo -n "threads=$t: "
  ./omp_count_sort 10000 $t
done
