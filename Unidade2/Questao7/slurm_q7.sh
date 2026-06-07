#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=saida_q7.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -p sequana_cpu_dev
#SBATCH -J omp_atomic_q7
#SBATCH --exclusive

module load gcc/13.2_sequana

gcc -fopenmp -o omp_atomic_test omp_atomic_test.c -lm

echo "=== 1 thread, n=10000000 ==="
./omp_atomic_test 10000000 1

echo "=== 2 threads, n=10000000 ==="
./omp_atomic_test 10000000 2

echo "=== 4 threads, n=10000000 ==="
./omp_atomic_test 10000000 4

echo "=== 8 threads, n=10000000 ==="
./omp_atomic_test 10000000 8
