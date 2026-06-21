/*
 * Questão 12 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Cronometrar a regra trapezoidal MPI para diferentes combinações
 * de (p = número de processos, n = número de trapézios), e reportar o
 * tempo de wall clock medido com MPI_Wtime.
 *
 * Conceito de wall clock paralelo:
 *   Cada processo mede seu próprio tempo. O tempo total da aplicação é
 *   o tempo do processo MAIS LENTO (gargalo), obtido com MPI_Reduce + MPI_MAX.
 *
 * Compile: mpicc -g -Wall -O2 -o mpi_trap_q12 mpi_trap_q12.c
 * Run:     mpiexec -n <p> ./mpi_trap_q12 <n>
 *
 * Exemplo: mpiexec -n 4 ./mpi_trap_q12 1000000
 */

#include <stdio.h>    /* printf, fprintf */
#include <stdlib.h>   /* atoi */
#include <mpi.h>

double Trap(double left, double right, int n, double h);
double f(double x);

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int n;             /* número de trapézios (argumento da linha de comando) */
    int local_n;       /* trapézios atribuídos a este processo */

    /* Intervalo fixo de integração [0, 1] */
    double a = 0.0, b = 1.0;
    double h;          /* largura de cada trapézio */
    double local_a, local_b;   /* subintervalo deste processo */
    double local_int;          /* integral local */
    double total_int;          /* soma das integrais (processo 0) */

    /* Variáveis de temporização */
    double start;        /* marca de início (segundos) */
    double finish;       /* marca de fim (segundos) */
    double elapsed;      /* tempo deste processo */
    double max_elapsed;  /* maior tempo entre todos os processos */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Valida argumento: n deve ser passado na linha de comando */
    if (argc < 2) {
        if (my_rank == 0) fprintf(stderr, "Uso: %s <n>\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    /* Converte o argumento string para inteiro */
    n = atoi(argv[1]);

    /* Distribui n para todos — embora argv[] seja acessível de todos os processos,
     * o broadcast garante consistência caso o MPI não propague argv. */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Calcula parâmetros do subintervalo de cada processo */
    h       = (b - a) / n;          /* largura de cada trapézio */
    local_n = n / comm_sz;           /* fatia igual para todos (n deve ser múltiplo de p) */
    local_a = a + my_rank * local_n * h;   /* início do subintervalo deste processo */
    local_b = local_a + local_n * h;       /* fim do subintervalo */

    /* ── TEMPORIZAÇÃO ── */

    /*
     * MPI_Barrier: todos os processos sincronizam ANTES de iniciar o timer.
     * Garante que nenhum processo começa a computação enquanto outro ainda
     * está na fase de setup, evitando diferenças artificiais de tempo.
     */
    MPI_Barrier(MPI_COMM_WORLD);

    /*
     * MPI_Wtime: retorna o tempo de wall clock em segundos (ponto flutuante).
     * Chamada antes da computação → marca de início.
     * É garantido que é monotonicamente crescente dentro de um processo.
     */
    start = MPI_Wtime();

    /* Computação: integração trapezoidal local */
    local_int = Trap(local_a, local_b, local_n, h);

    /*
     * MPI_Reduce: soma as integrais locais.
     * (Também consome tempo de comunicação, incluído na medição.)
     */
    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Marca de fim, após toda a computação E comunicação */
    finish  = MPI_Wtime();

    /* Tempo decorrido neste processo */
    elapsed = finish - start;

    /* ── COLETA DO TEMPO MÁXIMO ── */

    /*
     * MPI_Reduce com MPI_MAX: cada processo contribui com seu 'elapsed'.
     * O processo 0 recebe o MAIOR valor — este é o tempo real de wall clock
     * da aplicação paralela (o gargalo determina o tempo total).
     *
     * Por que MPI_MAX e não MPI_SUM?
     *   Os processos rodam em PARALELO. O tempo total é limitado pelo
     *   processo mais lento, não pela soma de todos os tempos.
     */
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Processo 0 imprime os resultados */
    if (my_rank == 0) {
        printf("p=%d n=%d integral=%.15e tempo=%.6f s\n",
               comm_sz, n, total_int, max_elapsed);
    }

    MPI_Finalize();
    return 0;
}

/* Regra trapezoidal composta no subintervalo [left, right] com n trapézios */
double Trap(double left, double right, int n, double h) {
    double est = (f(left) + f(right)) / 2.0;
    for (int i = 1; i <= n - 1; i++) est += f(left + i * h);
    return est * h;
}

/* Função a integrar: f(x) = x² → integral exata em [0,1] = 1/3 */
double f(double x) { return x * x; }

/*
 * INTERPRETAÇÃO DOS RESULTADOS:
 *
 * Para medir speedup S(p) e eficiência E(p):
 *   T(1)   = tempo com 1 processo (execução serial)
 *   T(p)   = max_elapsed com p processos
 *   S(p)   = T(1) / T(p)       → ganho de velocidade
 *   E(p)   = S(p) / p * 100%   → eficiência de uso dos processos
 *
 * Tabela típica (aumentando n e p):
 *   p=1, n=10^6: T≈X s
 *   p=2, n=10^6: T≈X/2 s (speedup ≈ 2)
 *   p=4, n=10^6: T≈X/4 s (speedup ≈ 4)
 *   → Problema embarrassingly parallel: speedup quase ideal.
 */
