# Programação Concorrente e Distribuída

Repositório com exercícios e implementações da disciplina de **Programação Concorrente e Distribuída (PCD)**.

## Conteúdo

### Unidade 2 — Memória Compartilhada com OpenMP

| Questão | Descrição | Arquivos |
|---------|-----------|----------|
| Q1 | Race condition em `omp_trap_1.c` sem `critical` | `resposta.txt`, `slurm_q1.sh` |
| Q2 | Cronometragem com `omp_get_wtime()` | `omp_trap_cronometrado.c`, `resposta.txt` |
| Q3 | Aritmética de ponto flutuante no Bleeblon | `resposta.txt` |
| Q4 | Escalonamento padrão de laços `parallel for` | `omp_escalonamento.c`, `slurm_q4.sh` |
| Q5 | Eliminação de dependência com fórmula fechada | `resposta.txt` |
| Q6 | `schedule(runtime)` e variável `OMP_SCHEDULE` | `omp_trap_schedule.c`, `slurm_q6.sh` |
| Q7 | Diretiva `atomic` com variáveis privadas | `omp_atomic_test.c`, `slurm_q7.sh` |
| Q8 | Perfilamento de cache com Valgrind/cachegrind | `resposta.txt` |
| Q9 | Falso compartilhamento em multiplicação matriz-vetor | `resposta.txt` |
| Q10 | Tokenizador thread-safe sem modificar a string | `tokenizador_threadsafe.c`, `resposta.txt` |
| Q11 | Histograma paralelo com histogramas locais | `omp_histograma.c`, `resposta.txt` |
| Q12 | Monte Carlo para estimar π | `omp_monte_carlo_pi.c`, `resposta.txt` |
| Q13 | Count Sort paralelo com análise de desempenho | `omp_count_sort.c`, `slurm_q13.sh`, `resposta.txt` |

## Compilação

```bash
# Exemplo geral (arquivo com OpenMP)
gcc -fopenmp -o programa programa.c -lm

# Questão 2
gcc -fopenmp -o omp_trap_cronometrado omp_trap_cronometrado.c -lm

# Questão 12 — Monte Carlo
gcc -fopenmp -o omp_monte_carlo_pi omp_monte_carlo_pi.c -lm
./omp_monte_carlo_pi 1000000000 4

# Questão 13 — Count Sort
gcc -fopenmp -O2 -o omp_count_sort omp_count_sort.c
./omp_count_sort 10000 8
```

## Execução no Supercomputador Santos Dumont (LNCC)

Os scripts `.sh` usam SLURM e estão configurados para a partição `sequana_cpu_dev`:

```bash
sbatch slurm_q1.sh
sbatch slurm_q4.sh
sbatch slurm_q6.sh
sbatch slurm_q7.sh
sbatch slurm_q13.sh
```

## Tecnologias

- **C** (C99/C11)
- **OpenMP** — paralelismo de memória compartilhada
- **SLURM** — gerenciador de jobs para HPC
- **Valgrind/cachegrind** — perfilamento de cache
