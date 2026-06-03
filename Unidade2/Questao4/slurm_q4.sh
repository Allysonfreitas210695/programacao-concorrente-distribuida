#!/bin/bash
#SBATCH --job-name=omp_escalonamento_q4
#SBATCH --output=saida_q4.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
#SBATCH --partition=sequana_cpu_dev

module load gcc
gcc -fopenmp -o omp_escalonamento omp_escalonamento.c

echo "=== 2 threads, 4 iteracoes ==="
./omp_escalonamento 4 2

echo "=== 4 threads, 8 iteracoes ==="
./omp_escalonamento 8 4

echo "=== 8 threads, 16 iteracoes ==="
./omp_escalonamento 16 8

echo "=== 16 threads, 100 iteracoes ==="
./omp_escalonamento 100 16
