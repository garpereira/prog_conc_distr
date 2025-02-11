#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 2000 // Tamanho da matriz
#define T 1000 // Número de iterações
#define D 0.1  // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

#define MAX_THREADS 8

void diff_eq(double **C, double **C_new, int local_rows, int rank, int size) {
    for (int t = 0; t < T; t++) {
        // Troca de bordas entre processos vizinhos
        if (rank > 0) {
            MPI_Sendrecv(C[1], N, MPI_DOUBLE, rank - 1, 0, C[0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(C[local_rows - 2], N, MPI_DOUBLE, rank + 1, 0, C[local_rows - 1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Cálculo da equação de difusão
        omp_set_num_threads(MAX_THREADS);
        #pragma omp parallel for shared(C_new)
        for (int i = 1; i < local_rows - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }

        // Atualização e cálculo da diferença média local
        double local_diff = 0.;
        #pragma omp parallel for shared(C) reduction(+:local_diff)
        for (int i = 1; i < local_rows - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                local_diff += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }

        // Redução da diferença média
        double global_diff;
        MPI_Reduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && (t % 100) == 0) {
            printf("Iteração %d - Diferença média = %g\n", t, global_diff / ((N - 2) * (N - 2)));
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rows = N / size + 2;  // Cada processo recebe +2 linhas para bordas

    // Alocação da matriz local
    double **C = (double **)malloc(local_rows * sizeof(double *));
    double **C_new = (double **)malloc(local_rows * sizeof(double *));
    for (int i = 0; i < local_rows; i++) {
        C[i] = (double *)malloc(N * sizeof(double));
        C_new[i] = (double *)malloc(N * sizeof(double));
    }

    // Inicialização
    omp_set_num_threads(MAX_THREADS);
    #pragma omp parallel for shared(C, C_new)
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            C_new[i][j] = 0.0;
        }
    }

    // Processo responsável pelo centro inicia a concentração
    if (rank == size / 2) {
        int local_center = (N / size) / 2;
        C[local_center][N / 2] = 1.0;
    }

    // Executa a equação de difusão
    diff_eq(C, C_new, local_rows, rank, size);

    // Coletar os resultados no processo mestre
    double *final_C = NULL;
    if (rank == 0) {
        final_C = (double *)malloc(N * N * sizeof(double));
    }

    MPI_Gather(&C[1][0], (N / size) * N, MPI_DOUBLE, final_C, (N / size) * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Processo imprime a concentração no centro
    if (rank == size / 2) {
        int local_center = (N / size) / 2;
        printf("Concentração final no centro: %f\n", C[local_center][N / 2]);
    }

    // Libera memória
    for (int i = 0; i < local_rows; i++) {
        free(C[i]);
        free(C_new[i]);
    }
    free(C);
    free(C_new);

    MPI_Finalize();
    return 0;
}
