/*
 * Questão 11 - Lista MPI (VERSÃO COMENTADA LINHA A LINHA)
 *
 * Objetivo: Reimplementar a função Get_input da regra trapezoidal
 * usando MPI_Pack e MPI_Unpack em vez de três chamadas MPI_Bcast.
 *
 * O que é MPI_Pack / MPI_Unpack?
 *   Permitem serializar dados de tipos DIFERENTES em um único buffer de bytes
 *   (char[]), transmiti-los como um bloco com MPI_PACKED, e desserializar
 *   no receptor. Útil quando se deseja enviar heterogêneos com uma só chamada.
 *
 * Neste caso: empacota um double (a), um double (b) e um int (n)
 * em 100 bytes, faz Bcast de uma vez, e cada processo desempacota.
 *
 * Compile: mpicc -g -Wall -o mpi_trap_q11 mpi_trap_q11.c
 * Run:     mpiexec -n <p> ./mpi_trap_q11
 */

#include <stdio.h>   /* printf, scanf */
#include <mpi.h>

double Trap(double left, double right, int n, double h);
double f(double x);
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p);

int main(void) {
    int my_rank, comm_sz, n, local_n;
    double a, b, h, local_a, local_b;
    double local_int, total_int;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Usa Get_input com Pack/Unpack (implementação abaixo) */
    Get_input(my_rank, comm_sz, &a, &b, &n);

    /* Calcula parâmetros locais — igual à versão original */
    h       = (b - a) / n;
    local_n = n / comm_sz;
    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;

    local_int = Trap(local_a, local_b, local_n, h);

    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
        printf("Integral de %f a %f com n=%d: %.15e\n", a, b, n, total_int);

    MPI_Finalize();
    return 0;
}

/*
 * Get_input — MODIFICAÇÃO PRINCIPAL: usa MPI_Pack/MPI_Unpack
 *
 * Fluxo:
 *   1. Processo 0 lê a, b, n do usuário.
 *   2. Processo 0 empacota os três valores em pack_buf[].
 *   3. MPI_Bcast distribui pack_buf para todos (tipo MPI_PACKED).
 *   4. Processos != 0 desempacotam pack_buf para recuperar a, b, n.
 */
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p) {
    /* Buffer de bytes que conterá os dados serializados.
     * 100 bytes é suficiente para 2 doubles (8 bytes cada) + 1 int (4 bytes) + overhead. */
    char pack_buf[100];

    /* 'position' rastreia o cursor de escrita/leitura dentro de pack_buf.
     * Deve ser inicializado em 0 antes de empacotar. */
    int position = 0;

    if (my_rank == 0) {
        printf("Entre com a, b e n:\n");
        scanf("%lf %lf %d", a_p, b_p, n_p);

        /*
         * MPI_Pack: serializa 'a' no buffer pack_buf a partir de 'position'.
         *
         * Parâmetros:
         *   a_p         → ponteiro para o dado a empacotar
         *   1           → número de elementos a empacotar
         *   MPI_DOUBLE  → tipo do dado
         *   pack_buf    → buffer de destino (bytes)
         *   100         → tamanho total do buffer (em bytes)
         *   &position   → posição atual no buffer; ATUALIZADO após chamada
         *   MPI_COMM_WORLD → comunicador (necessário para padding/alinhamento)
         *
         * Após cada MPI_Pack, 'position' avança para logo após o dado serializado.
         * Chamadas sequenciais concatenam os dados sem espaços.
         */
        MPI_Pack(a_p, 1, MPI_DOUBLE, pack_buf, 100, &position, MPI_COMM_WORLD);

        /* Empacota 'b' logo após 'a' no buffer */
        MPI_Pack(b_p, 1, MPI_DOUBLE, pack_buf, 100, &position, MPI_COMM_WORLD);

        /* Empacota 'n' logo após 'b' no buffer */
        MPI_Pack(n_p, 1, MPI_INT,    pack_buf, 100, &position, MPI_COMM_WORLD);

        /* Após as três chamadas, pack_buf[0..position-1] contém a, b, n serializados */
    }

    /*
     * MPI_Bcast com tipo MPI_PACKED:
     *   Transmite os 100 bytes do buffer de dados empacotados para todos.
     *   Usamos o tamanho fixo 100 (não apenas os bytes usados) para simplificar.
     *   MPI_PACKED é o tipo especial que sinaliza ao MPI que o buffer
     *   contém dados empacotados com MPI_Pack.
     */
    MPI_Bcast(pack_buf, 100, MPI_PACKED, 0, MPI_COMM_WORLD);

    if (my_rank != 0) {
        /* Reinicia o cursor para leitura a partir do início do buffer */
        position = 0;

        /*
         * MPI_Unpack: desserializa 'a' do buffer pack_buf.
         *
         * Parâmetros:
         *   pack_buf    → buffer de origem (bytes recebidos)
         *   100         → tamanho total do buffer
         *   &position   → posição atual; ATUALIZADO após chamada
         *   a_p         → onde armazenar o dado desserializado
         *   1           → número de elementos a desserializar
         *   MPI_DOUBLE  → tipo esperado
         *   MPI_COMM_WORLD
         *
         * ATENÇÃO: a ordem de desempacotamento deve ser IDÊNTICA à ordem
         * de empacotamento: primeiro a, depois b, depois n.
         */
        MPI_Unpack(pack_buf, 100, &position, a_p, 1, MPI_DOUBLE, MPI_COMM_WORLD);

        /* Desempacota 'b' logo após 'a' */
        MPI_Unpack(pack_buf, 100, &position, b_p, 1, MPI_DOUBLE, MPI_COMM_WORLD);

        /* Desempacota 'n' logo após 'b' */
        MPI_Unpack(pack_buf, 100, &position, n_p, 1, MPI_INT,    MPI_COMM_WORLD);
    }
    /* Agora todos os processos têm a_p, b_p, n_p com os valores corretos */
}

/* Regra trapezoidal: aproximação numérica da integral de f em [left, right] */
double Trap(double left, double right, int n, double h) {
    double est = (f(left) + f(right)) / 2.0;
    for (int i = 1; i <= n - 1; i++) est += f(left + i * h);
    return est * h;
}

/* Função a integrar: f(x) = x² */
double f(double x) { return x * x; }

/*
 * COMPARAÇÃO: versão original vs. versão com Pack
 *
 * Original (3 broadcasts):
 *   MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, comm)
 *   MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, comm)
 *   MPI_Bcast(n_p, 1, MPI_INT,    0, comm)
 *   → 3 mensagens na rede.
 *
 * Com Pack (1 broadcast):
 *   MPI_Pack(...a) → MPI_Pack(...b) → MPI_Pack(...n)
 *   MPI_Bcast(pack_buf, 100, MPI_PACKED, 0, comm)
 *   MPI_Unpack(...a) → MPI_Unpack(...b) → MPI_Unpack(...n)
 *   → 1 mensagem na rede (latência reduzida).
 */
