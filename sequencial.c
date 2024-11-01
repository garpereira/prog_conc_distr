#include <stdio.h>

#define N 100 // Tamanho da grade
#define T 1000 // Número de iterações
#define D 0.1 // Coeficiente de difusão
#define DELTA_T 0.01 // Passo de tempo
#define DELTA_X 1.0 // Passo de espaço

void diff_eq(double C[N][N], double C_new[N][N]){
    for (int t = 0; t < T; t++){
        for (int i = 1; i < N - 1; i++){
            for (int j = 1; j < N - 1; j++){
                C_new[i][j] = C[i][j] + D * DELTA_T / (DELTA_X * DELTA_X) * (C[i + 1][j] + C[i - 1][j] + C[i][j + 1] + C[i][j - 1] - 4 * C[i][j]);
            }
        }
        // Atualizar a matriz C para a próxima iteração
        for (int i = 1; i < N - 1; i++){
            for (int j = 1; j < N - 1; j++)
                C[i][j] = C_new[i][j];
        }
    }
}

int main(int argc, char **argv){
    double C[N][N] = {0}; // Concentração Inicial
    double C_new[N][N] = {0}; // Concentração para a próxima iteração

    // Inicializar uma concentraçãoalta no centro
    C[N/2][N/2] = 1.0;

    // Executar a equação de difusão
    diff_eq(C, C_new);

    // Exibir resultado para verificação
    printf("Concentração final no centro: %f\n", C[N/2][N/2]);
}