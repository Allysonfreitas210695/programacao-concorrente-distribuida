#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

/*
 * Tokenizador thread-safe que não modifica a string de entrada.
 * Retorna o próximo token em 'buffer' e atualiza '*pos'.
 * Retorna 1 se encontrou token, 0 se chegou ao fim.
 */
int proximo_token(const char* str, int* pos, char* buffer, int buf_size) {
    int i = *pos;

    while (str[i] != '\0' && isspace((unsigned char)str[i]))
        i++;

    if (str[i] == '\0') {
        *pos = i;
        return 0;
    }

    int j = 0;
    while (str[i] != '\0' && !isspace((unsigned char)str[i]) && j < buf_size - 1) {
        buffer[j++] = str[i++];
    }
    buffer[j] = '\0';
    *pos = i;
    return 1;
}

int main() {
    const char* texto = "OpenMP e incrivel para programacao paralela em C";
    char token[256];

    #pragma omp parallel num_threads(4)
    {
        int pos = 0;
        int tid = omp_get_thread_num();

        if (tid == 0) {
            printf("Tokens encontrados:\n");
            while (proximo_token(texto, &pos, token, sizeof(token))) {
                printf("  '%s'\n", token);
            }
        }
    }

    printf("\nUso com strings independentes por thread:\n");
    const char* strings[] = {
        "thread zero processa isto",
        "thread um processa aquilo",
        "thread dois processa esse texto",
        "thread tres processa outro texto"
    };

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        int pos = 0;
        char tok[256];
        printf("Thread %d tokens: ", tid);
        while (proximo_token(strings[tid], &pos, tok, sizeof(tok))) {
            printf("[%s] ", tok);
        }
        printf("\n");
    }

    return 0;
}
