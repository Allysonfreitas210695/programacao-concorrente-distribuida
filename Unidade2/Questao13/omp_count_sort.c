/* Inclui funções de entrada/saída padrão (printf) */
#include <stdio.h>
/* Inclui funções de alocação de memória (malloc, free) e atoi */
#include <stdlib.h>
/* Inclui memcpy para cópia de blocos de memória */
#include <string.h>
/* Inclui a API OpenMP */
#include <omp.h>

/*
 * Função: Count_sort_paralelo
 * ---------------------------
 * Ordena o array 'a' de tamanho 'n' usando Count Sort paralelizado com OpenMP.
 *
 * Ideia: para cada elemento a[i], conta quantos elementos são menores que ele.
 * Esse contador (count) é a posição final de a[i] no array ordenado.
 * Empates são resolvidos pelo índice j: se a[j] == a[i] e j < i, conta como menor.
 *
 * O laço externo (for i) é paralelizado — cada iteração é independente
 * pois calcula count apenas lendo a[] e escreve em temp[count] único.
 */
void Count_sort_paralelo(int a[], int n, int thread_count) {
    /* Aloca array temporário onde os elementos serão inseridos em ordem */
    int* temp = (int*)malloc(n * sizeof(int));

    /* Paraleliza o laço externo entre thread_count threads.
     * schedule(static) divide as iterações em blocos contíguos iguais,
     * distribuídas antes da execução — menor overhead de escalonamento */
    #pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int i = 0; i < n; i++) {
        /* count: quantos elementos devem vir antes de a[i] no resultado final */
        int count = 0;

        /* Laço interno: compara a[i] com todos os outros elementos */
        for (int j = 0; j < n; j++) {
            /* Se a[j] é estritamente menor que a[i], incrementa count */
            if (a[j] < a[i])
                count++;
            /* Se a[j] == a[i] e j < i, trata a[j] como "menor" para desempate.
             * Isso garante posições únicas mesmo com valores repetidos */
            else if (a[j] == a[i] && j < i)
                count++;
        }
        /* Insere a[i] na posição exata calculada pelo count */
        temp[count] = a[i];
    }
    /* Barreira implícita: todas as threads terminam antes de continuar */

    /* Copia o array ordenado (temp) de volta para a[] em paralelo.
     * Seguro de paralelizar: cada posição i é escrita por exatamente 1 thread */
    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < n; i++)
        a[i] = temp[i]; /* Sobrescreve a posição i com o valor ordenado */

    /* Libera o array temporário que já foi copiado de volta para a[] */
    free(temp);
}

/*
 * Função: Count_sort_serial
 * -------------------------
 * Versão serial do Count Sort — usada para comparação de desempenho.
 * Mesma lógica da versão paralela, mas executada por uma única thread.
 */
void Count_sort_serial(int a[], int n) {
    /* Aloca array temporário para receber os elementos em ordem */
    int* temp = (int*)malloc(n * sizeof(int));

    /* Para cada elemento a[i], calcula sua posição final */
    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (a[j] < a[i]) count++;             /* a[j] vem antes de a[i] */
            else if (a[j] == a[i] && j < i) count++; /* desempate por índice */
        }
        temp[count] = a[i]; /* Coloca a[i] na posição correta */
    }

    /* Copia o array ordenado de temp de volta para a[].
     * n*sizeof(int): número total de bytes a copiar */
    memcpy(a, temp, n * sizeof(int));

    /* Libera a memória temporária */
    free(temp);
}

/*
 * Função: cmp
 * -----------
 * Função de comparação para qsort (biblioteca padrão C).
 * Retorna negativo se *x < *y, zero se igual, positivo se *x > *y.
 * Usada como referência de desempenho para comparar com Count Sort.
 */
int cmp(const void* x, const void* y) {
    /* Cast de void* para int* e subtração: forma clássica de comparador para qsort */
    return (*(int*)x - *(int*)y);
}

int main(int argc, char* argv[]) {
    /* Verifica se os argumentos obrigatórios foram passados */
    if (argc < 3) {
        printf("Uso: %s <n> <num_threads>\n", argv[0]);
        return 1; /* Encerra com código de erro */
    }

    /* n: tamanho do array a ser ordenado */
    int n            = atoi(argv[1]);

    /* thread_count: número de threads OpenMP para a versão paralela */
    int thread_count = atoi(argv[2]);

    /* Aloca 3 cópias idênticas do array para testar os 3 algoritmos separadamente */
    int* a1 = (int*)malloc(n * sizeof(int)); /* Para Count Sort serial    */
    int* a2 = (int*)malloc(n * sizeof(int)); /* Para Count Sort paralelo  */
    int* a3 = (int*)malloc(n * sizeof(int)); /* Para qsort (referência)   */

    /* Inicializa o gerador com semente fixa para resultados reproduzíveis */
    srand(42);

    /* Preenche os 3 arrays com os mesmos valores aleatórios no intervalo [0, 2n-1] */
    for (int i = 0; i < n; i++)
        a1[i] = a2[i] = a3[i] = rand() % (n * 2); /* Mesmo valor nos 3 arrays */

    /* Variáveis para cronometrar cada algoritmo */
    double t_inicio, t_fim;

    /* ----- TESTE 1: Count Sort serial ----- */
    t_inicio = omp_get_wtime();             /* Marca início */
    Count_sort_serial(a1, n);              /* Ordena a1 serialmente */
    t_fim = omp_get_wtime();               /* Marca fim */
    printf("Count sort serial  : %.4f segundos\n", t_fim - t_inicio);

    /* ----- TESTE 2: Count Sort paralelo ----- */
    t_inicio = omp_get_wtime();
    Count_sort_paralelo(a2, n, thread_count); /* Ordena a2 com OpenMP */
    t_fim = omp_get_wtime();
    printf("Count sort paralelo: %.4f segundos (%d threads)\n",
           t_fim - t_inicio, thread_count);

    /* ----- TESTE 3: qsort da stdlib (Quicksort O(n log n)) ----- */
    t_inicio = omp_get_wtime();
    qsort(a3, n, sizeof(int), cmp); /* Ordena a3 com a função da stdlib */
    t_fim = omp_get_wtime();
    printf("qsort (serial)     : %.4f segundos\n", t_fim - t_inicio);

    /* ----- VERIFICAÇÃO DE CORRETUDE ----- */
    /* Compara o resultado do Count Sort paralelo com o do qsort elemento a elemento.
     * Se diferir em qualquer posição, o algoritmo está com bug */
    int correto = 1; /* Assume correto até encontrar divergência */
    for (int i = 0; i < n; i++)
        if (a2[i] != a3[i]) { correto = 0; break; } /* Marca como incorreto e para */

    /* Imprime se o resultado paralelo é correto ou não */
    printf("Resultado paralelo %s\n", correto ? "CORRETO" : "INCORRETO");

    /* Libera os 3 arrays da memória */
    free(a1); free(a2); free(a3);

    return 0; /* Programa encerrado com sucesso */
}
