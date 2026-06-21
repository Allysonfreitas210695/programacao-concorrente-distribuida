/*
 * Questão 1 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Modificar a regra trapezoidal MPI para funcionar corretamente
 * mesmo quando o número de trapézios (n) NÃO é divisível pelo número de
 * processos (comm_sz). A versão original assumia n % comm_sz == 0.
 *
 * Compile: mpicc -g -Wall -o mpi_trap_q1 mpi_trap_q1.c
 * Run:     mpiexec -n <p> ./mpi_trap_q1
 */

#include <stdio.h>   /* printf, scanf */
#include <mpi.h>     /* todas as funções MPI_ */

/* Protótipos das funções definidas abaixo do main */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len);
double f(double x);
void Get_input(int my_rank, double* a_p, double* b_p, int* n_p);

int main(void) {
    /* my_rank  : identificador do processo atual (0 a comm_sz-1) */
    /* comm_sz  : número total de processos no comunicador         */
    /* n        : número total de trapézios solicitado pelo usuário */
    /* local_n  : número de trapézios que ESTE processo calculará  */
    int my_rank, comm_sz, n, local_n;

    /* a, b      : extremos globais do intervalo de integração     */
    /* h         : largura de cada trapézio = (b-a)/n              */
    /* local_a   : extremo esquerdo do subintervalo deste processo */
    /* local_b   : extremo direito  do subintervalo deste processo */
    double a, b, h, local_a, local_b;

    /* local_int  : resultado da integração local deste processo   */
    /* total_int  : soma de todas as integrais locais (somente p0) */
    double local_int, total_int;

    /* Inicializa o ambiente MPI. NULL, NULL = sem argumentos especiais. */
    MPI_Init(NULL, NULL);

    /* Obtém o rank (id) do processo atual dentro de MPI_COMM_WORLD */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Obtém o número total de processos em MPI_COMM_WORLD */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Processo 0 lê a, b, n do usuário e faz broadcast para todos */
    Get_input(my_rank, &a, &b, &n);

    /* Calcula a largura de cada trapézio */
    h = (b - a) / n;

    /*
     * MODIFICAÇÃO PRINCIPAL em relação à versão original:
     *
     * Problema anterior: local_n = n / comm_sz (divisão inteira)
     *   → se n=10 e comm_sz=3, dois processos ficam com 3 e um fica sem o trapézio extra.
     *
     * Solução: distribuir o resto (remainder) entre os primeiros processos.
     *   base_n    = parte inteira da divisão → cada processo recebe pelo menos isso.
     *   remainder = sobra após divisão       → primeiros 'remainder' processos ganham +1.
     */
    int base_n    = n / comm_sz;   /* quantos trapézios cada um recebe no mínimo */
    int remainder = n % comm_sz;   /* trapézios "extras" a distribuir */

    if (my_rank < remainder) {
        /* Este processo é um dos que recebem +1 trapézio */
        local_n = base_n + 1;
        /* O intervalo local começa após todos os trapézios dos processos anteriores.
         * Como os processos 0..my_rank-1 também têm (base_n+1) trapézios cada,
         * o deslocamento é my_rank * local_n trapézios. */
        local_a = a + my_rank * local_n * h;
    } else {
        /* Este processo recebe apenas base_n trapézios */
        local_n = base_n;
        /* Os 'remainder' primeiros processos já ocuparam (remainder * (base_n+1)) trapézios.
         * Este processo começa após eles + os (my_rank - remainder) processos anteriores
         * a ele que também têm base_n trapézios cada. */
        local_a = a + (my_rank * base_n + remainder) * h;
    }

    /* Extremo direito do subintervalo deste processo */
    local_b = local_a + local_n * h;

    /* Aplica a regra trapezoidal no subintervalo local */
    local_int = Trap(local_a, local_b, local_n, h);

    /*
     * MPI_Reduce: soma todas as integrais locais em total_int no processo 0.
     * Parâmetros:
     *   &local_int  → dado de entrada de cada processo
     *   &total_int  → onde o resultado vai (só no processo raiz)
     *   1           → quantidade de elementos
     *   MPI_DOUBLE  → tipo do dado
     *   MPI_SUM     → operação de redução (soma)
     *   0           → processo raiz que recebe o resultado
     *   MPI_COMM_WORLD → comunicador
     */
    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Apenas o processo 0 imprime o resultado final */
    if (my_rank == 0) {
        printf("Com n = %d trapézios, estimativa da integral de %f a %f = %.15e\n",
               n, a, b, total_int);
    }

    /* Finaliza o ambiente MPI — deve ser a última chamada MPI */
    MPI_Finalize();
    return 0;
}

/*
 * Get_input: processo 0 lê os valores e os distribui via broadcast.
 * Broadcast é necessário pois apenas o processo 0 tem acesso ao stdin.
 */
void Get_input(int my_rank, double* a_p, double* b_p, int* n_p) {
    if (my_rank == 0) {
        printf("Entre com a, b e n:\n");
        scanf("%lf %lf %d", a_p, b_p, n_p);   /* lê do terminal */
    }
    /* Envia 'a' de proc 0 para todos os outros */
    MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* Envia 'b' de proc 0 para todos os outros */
    MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* Envia 'n' de proc 0 para todos os outros */
    MPI_Bcast(n_p, 1, MPI_INT,    0, MPI_COMM_WORLD);
}

/*
 * Trap: calcula a integral da função f no intervalo [left_endpt, right_endpt]
 * usando a regra trapezoidal composta com 'trap_count' trapézios de largura 'base_len'.
 *
 * Fórmula: h * [f(x0)/2 + f(x1) + f(x2) + ... + f(x_{n-1}) + f(xn)/2]
 */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len) {
    double estimate, x;
    int i;

    /* Caso degenerado: nenhum trapézio a calcular */
    if (trap_count == 0) return 0.0;

    /* Contribuição das extremidades com peso 1/2 cada */
    estimate = (f(left_endpt) + f(right_endpt)) / 2.0;

    /* Pontos internos têm peso inteiro 1 */
    for (i = 1; i <= trap_count - 1; i++) {
        x = left_endpt + i * base_len;   /* x_i = a + i*h */
        estimate += f(x);
    }

    /* Multiplica pelo passo h para obter a área total */
    return estimate * base_len;
}

/* Função a integrar: f(x) = x² */
double f(double x) { return x * x; }
