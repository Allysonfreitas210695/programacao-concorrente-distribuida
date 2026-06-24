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
#SBATCH --time=00:20:00

module load openmpi/gnu/4.1.1

# Compila versão com temporização (sem impressão do vetor)
mpicc -O2 -o mpi_odd_even_timing mpi_odd_even_timing.c

# n = 24000: divisível por 1,2,4,8,16,24,48,96
N=24000

# Arquivo de saída com os resultados em texto
RES=resultados_q13.txt
{
    echo "Resultados Questão 13 - Ordenação Ímpar-Par (n=$N)"
    echo "Job: $SLURM_JOB_ID   Data: $(date)"
    echo "========================================================"

    echo "=== 1 nó, variando p ==="
    for P in 1 2 4 8 16 24; do
        srun --nodes=1 --ntasks=$P ./mpi_odd_even_timing $N
    done

    echo "=== 2 nós completos (p=48) ==="
    srun --nodes=2 --ntasks=48 ./mpi_odd_even_timing $N

    echo "=== 4 nós completos (p=96) ==="
    srun --nodes=4 --ntasks=96 ./mpi_odd_even_timing $N
} | tee "$RES"
