#!/bin/bash
#SBATCH --job-name=omp_atomic_q7
#SBATCH --output=saida_q7.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --partition=sequana_cpu_dev

module load gcc
gcc -fopenmp -o omp_atomic_test omp_atomic_test.c -lm

echo "=== n=10000000 ==="
./omp_atomic_test 10000000 4
./omp_atomic_test 10000000 8
./omp_atomic_test 10000000 16
