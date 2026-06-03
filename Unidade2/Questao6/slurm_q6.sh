#!/bin/bash
#SBATCH --job-name=omp_schedule_q6
#SBATCH --output=saida_q6.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --partition=sequana_cpu_dev

module load gcc
gcc -fopenmp -o omp_trap_schedule omp_trap_schedule.c -lm

echo "=== static ==="
OMP_SCHEDULE="static" ./omp_trap_schedule 16 4

echo "=== static,2 ==="
OMP_SCHEDULE="static,2" ./omp_trap_schedule 16 4

echo "=== dynamic ==="
OMP_SCHEDULE="dynamic" ./omp_trap_schedule 16 4

echo "=== dynamic,2 ==="
OMP_SCHEDULE="dynamic,2" ./omp_trap_schedule 16 4

echo "=== guided ==="
OMP_SCHEDULE="guided" ./omp_trap_schedule 16 4
