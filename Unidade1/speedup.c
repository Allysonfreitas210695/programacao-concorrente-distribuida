#include <stdio.h>
#include <math.h>

int main() {
    int n_vals[] = {10, 20, 40, 80, 160, 320};
    int p_vals[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int nn = 6, np = 8;

    printf("%-6s %-6s %-14s %-14s\n", "n", "p", "Speedup", "Eficiencia");

    for (int i = 0; i < nn; i++) {
        double n = n_vals[i];
        for (int j = 0; j < np; j++) {
            double p = p_vals[j];
            double Tseq = n * n;
            double Tpar = (n * n) / p + log2(p);
            double S    = Tseq / Tpar;
            double E    = S / p;
            printf("%-6.0f %-6.0f %-14.4f %-14.4f\n", n, p, S, E);
        }
    }
    return 0;
}