#!/bin/bash
# Script SLURM - Questão 13: Speedup/Eficiência da ordenação ímpar-par
# Santos Dumont: nó Sequana tem 24 núcleos.
# Aloca 4 nós (96 tarefas) para cobrir todos os casos do enunciado.

#SBATCH --job-name=odd_even_q13
#SBATCH --output=odd_even_q13_%j.out
#SBATCH --error=odd_even_q13_%j.err
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=4
#SBATCH --ntasks=96
#SBATCH --time=00:30:00

module load openmpi/gnu/4.1.1

# Copia o código fonte do diretório do livro e compila
cp ../../../ipp-source/ipp-source-use/ch3/mpi_odd_even.c .
mpicc -O2 -o mpi_odd_even mpi_odd_even.c

# n = 24000: divisível por 1,2,4,8,16,24,48,96
# Grande o suficiente para medir tempo com p pequeno,
# mas cabe em memória com p grande.
N=24000

echo "=== 1 nó, variando p ==="
for P in 1 2 4 8 16 24; do
    srun --nodes=1 --ntasks=$P ./mpi_odd_even g $N
done

echo "=== 2 nós completos (p=48) ==="
srun --nodes=2 --ntasks=48 ./mpi_odd_even g $N

echo "=== 4 nós completos (p=96) ==="
srun --nodes=4 --ntasks=96 ./mpi_odd_even g $N
