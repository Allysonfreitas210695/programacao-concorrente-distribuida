/* Inclui funções de entrada/saída padrão (printf) */
#include <stdio.h>
/* Inclui funções de alocação de memória */
#include <stdlib.h>
/* Inclui funções de manipulação de strings (strlen, etc.) */
#include <string.h>
/* Inclui funções de classificação de caracteres (isspace) */
#include <ctype.h>
/* Inclui a API OpenMP */
#include <omp.h>

/*
 * Função: proximo_token
 * ---------------------
 * Extrai o próximo token (palavra) da string 'str' a partir da posição '*pos'.
 * É thread-safe pois NÃO usa estado global nem modifica a string original.
 * Todo o estado fica em variáveis locais ou no ponteiro 'pos' passado pelo chamador.
 *
 * Parâmetros:
 *   str      - string de entrada (apenas leitura, nunca modificada)
 *   pos      - ponteiro para a posição atual; atualizado após cada chamada
 *   buffer   - onde o token encontrado é copiado
 *   buf_size - tamanho máximo do buffer (protege contra overflow)
 *
 * Retorno: 1 se encontrou um token, 0 se chegou ao fim da string
 */
int proximo_token(const char* str, int* pos, char* buffer, int buf_size) {
    /* Copia o valor atual de *pos para variável local i (posição de leitura) */
    int i = *pos;

    /* Pula todos os caracteres de espaço em branco (espaço, \t, \n, etc.)
     * antes do próximo token. isspace espera unsigned char para evitar UB */
    while (str[i] != '\0' && isspace((unsigned char)str[i]))
        i++; /* Avança um caractere por vez */

    /* Se chegou ao fim da string sem encontrar nada, sinaliza fim */
    if (str[i] == '\0') {
        *pos = i;  /* Atualiza a posição do chamador */
        return 0;  /* Retorna 0 indicando que não há mais tokens */
    }

    /* j é o índice de escrita no buffer do token */
    int j = 0;

    /* Copia caracteres para o buffer até encontrar espaço, fim da string,
     * ou estourar o tamanho do buffer (buf_size - 1 para reservar '\0') */
    while (str[i] != '\0' && !isspace((unsigned char)str[i]) && j < buf_size - 1) {
        buffer[j++] = str[i++]; /* Copia str[i] para buffer[j] e avança ambos */
    }

    /* Finaliza o token com o terminador nulo, tornando-o uma string C válida */
    buffer[j] = '\0';

    /* Atualiza a posição do chamador para após o token recém-extraído */
    *pos = i;

    return 1; /* Retorna 1 indicando que um token foi encontrado */
}

int main() {
    /* String de teste compartilhada — apenas lida, nunca modificada */
    const char* texto = "OpenMP e incrivel para programacao paralela em C";

    /* Buffer para receber tokens individualmente (máximo 255 chars + '\0') */
    char token[256];

    /* ----------------------------------------------------------------
     * DEMO 1: Apenas a thread 0 tokeniza o texto compartilhado.
     * As demais threads existem mas ficam ociosas no bloco if (tid != 0).
     * ---------------------------------------------------------------- */
    #pragma omp parallel num_threads(4)
    {
        /* pos é local a cada thread (declarado dentro do bloco paralelo) */
        int pos = 0;

        /* Obtém o identificador desta thread */
        int tid = omp_get_thread_num();

        /* Somente a thread 0 realiza a tokenização para evitar duplicatas
         * (as 4 threads começam na posição 0, então tokenizariam igual) */
        if (tid == 0) {
            printf("Tokens encontrados:\n");
            /* Chama proximo_token em loop até retornar 0 (fim da string).
             * 'pos' é atualizado a cada chamada, avançando pela string */
            while (proximo_token(texto, &pos, token, sizeof(token))) {
                printf("  '%s'\n", token); /* Imprime cada token extraído */
            }
        }
    } /* Barreira implícita: todas as 4 threads terminam aqui */

    /* ----------------------------------------------------------------
     * DEMO 2: Cada thread tokeniza uma string própria e independente.
     * Aqui o paralelismo é real: 4 threads, 4 strings diferentes.
     * ---------------------------------------------------------------- */
    printf("\nUso com strings independentes por thread:\n");

    /* Array com uma string para cada uma das 4 threads */
    const char* strings[] = {
        "thread zero processa isto",
        "thread um processa aquilo",
        "thread dois processa esse texto",
        "thread tres processa outro texto"
    };

    #pragma omp parallel num_threads(4)
    {
        /* Cada thread lê seu próprio identificador */
        int tid = omp_get_thread_num();

        /* pos é local a cada thread — começa em 0 para a string desta thread */
        int pos = 0;

        /* Buffer privado por thread (na pilha): sem risco de corrida */
        char tok[256];

        /* Exibe o cabeçalho com o número da thread */
        printf("Thread %d tokens: ", tid);

        /* Cada thread tokeniza strings[tid] — sua string exclusiva.
         * Como pos e tok são locais, não há compartilhamento nem corrida */
        while (proximo_token(strings[tid], &pos, tok, sizeof(tok))) {
            printf("[%s] ", tok); /* Imprime cada token desta thread */
        }
        printf("\n"); /* Quebra de linha após todos os tokens desta thread */
    } /* Barreira implícita: todas as threads terminam antes de continuar */

    return 0; /* Programa encerrado com sucesso */
}
