#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -p sequana_cpu_dev
#SBATCH -J omp_trap_q1
#SBATCH --output=Questao_01.txt
#SBATCH --exclusive

cd /scratch/pex1272-ufersa/allyson.fernandes/Questao_01

module load gcc/13.2_sequana

gcc -fopenmp -o omp_trap_1 omp_trap_1.c -lm

for threads in 1 2 4 8 16; do
 for n in 1000 10000 100000 1000000; do
   export OMP_NUM_THREADS=$threads
   echo "Threads=$threads, n=$n:"
   echo "0.0 3.0 $n" | ./omp_trap_1 $threads
 done
done
