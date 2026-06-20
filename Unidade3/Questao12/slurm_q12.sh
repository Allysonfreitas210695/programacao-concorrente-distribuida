#!/bin/bash
# Script SLURM - Questão 12: Cronometragem da Regra Trapezoidal
# Santos Dumont: nó Sequana tem 24 núcleos.
# Aloca 4 nós (96 tarefas) para cobrir todos os casos do enunciado.

#SBATCH --job-name=trap_q12
#SBATCH --output=trap_q12_%j.out
#SBATCH --error=trap_q12_%j.err
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=4
#SBATCH --ntasks=96
#SBATCH --time=00:30:00

module load openmpi/gnu/4.1.1

# Compila
mpicc -O2 -o mpi_trap_q12 mpi_trap_q12.c

# n = 1.000.000.000 (1 bilhão): garante que o trabalho computacional domine
# o overhead de comunicação (MPI_Reduce = O(log p)) para todos os p testados.
# Divisível por 1,2,4,8,16,24,48,96.
N=1000000000

echo "=== 1 nó, variando p ==="
for P in 1 2 4 8 16 24; do
    srun --nodes=1 --ntasks=$P ./mpi_trap_q12 $N
done

echo "=== 2 nós completos (p=48) ==="
srun --nodes=2 --ntasks=48 ./mpi_trap_q12 $N

echo "=== 4 nós completos (p=96) ==="
srun --nodes=4 --ntasks=96 ./mpi_trap_q12 $N
