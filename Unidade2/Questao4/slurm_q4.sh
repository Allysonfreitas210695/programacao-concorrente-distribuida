#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=omp_escalonamento_q4
#SBATCH --output=saida_q4.txt
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 
#SBATCH -p sequana_cpu_dev
#SBATCH --exclusive

module load gcc/13.2_sequana

gcc -fopenmp -o omp_escalonamento omp_escalonamento.c

echo "=== 2 threads, 4 iteracoes ==="
./omp_escalonamento 4 2

echo "=== 4 threads, 8 iteracoes ==="
./omp_escalonamento 8 4

echo "=== 8 threads, 16 iteracoes ==="
./omp_escalonamento 16 8

echo "=== 16 threads, 100 iteracoes ==="
./omp_escalonamento 100 16
