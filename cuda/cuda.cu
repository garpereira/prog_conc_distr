#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


#define N 2000 // Tamanho da Grade
#define T 1000 // Numero de Iterações
#define D 0.1 // Coeficiente de Difusão
#define DELTA_T 0.01 // Passo de Tempo
#define DELTA_X 1.0 // Passo de Espaço
#define RADIUS 1 // Raio 
#define BLOCK_SIZE 8 // Tamanho do Bloco

__global__ void IniciarMatriz(double *matriz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < N){
        matriz[i * N + j] = 0.;
    }
    __syncthreads();

}

__device__ double atomicAdd_double(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}


__global__ void diff_eq(double *matriz_in, double *matriz_new, double *difmedio_device) {
    __shared__ double sharedMem[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    int globalX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int globalY = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int localX = threadIdx.x + RADIUS;
    int localY = threadIdx.y + RADIUS;

    // Carregar dados para a memória compartilhada
    if (globalX < N && globalY < N) {
        sharedMem[localY][localX] = matriz_in[globalY * N + globalX];
    } else {
        sharedMem[localY][localX] = 0.0;
    }

    // Carregar bordas
    if (threadIdx.x < RADIUS) {
        if (globalX >= RADIUS) {
            sharedMem[localY][localX - RADIUS] = matriz_in[globalY * N + globalX - RADIUS];
        } else {
            sharedMem[localY][localX - RADIUS] = 0.0;
        }

        if (globalX + BLOCK_SIZE < N) {
            sharedMem[localY][localX + BLOCK_SIZE] = matriz_in[globalY * N + globalX + BLOCK_SIZE];
        } else {
            sharedMem[localY][localX + BLOCK_SIZE] = 0.0;
        }
    }

    if (threadIdx.y < RADIUS) {
        if (globalY >= RADIUS) {
            sharedMem[localY - RADIUS][localX] = matriz_in[(globalY - RADIUS) * N + globalX];
        } else {
            sharedMem[localY - RADIUS][localX] = 0.0;
        }

        if (globalY + BLOCK_SIZE < N) {
            sharedMem[localY + BLOCK_SIZE][localX] = matriz_in[(globalY + BLOCK_SIZE) * N + globalX];
        } else {
            sharedMem[localY + BLOCK_SIZE][localX] = 0.0;
        }
    }

    __syncthreads();

    double local_diff = 0.0;

    if (globalX < N && globalY < N && globalX >= RADIUS && globalX < N - RADIUS && globalY >= RADIUS && globalY < N - RADIUS) {
        double new_val = sharedMem[localY][localX] + D * DELTA_T * (
            (sharedMem[localY + 1][localX] + sharedMem[localY - 1][localX] +
             sharedMem[localY][localX + 1] + sharedMem[localY][localX - 1] -
             4 * sharedMem[localY][localX]) / (DELTA_X * DELTA_X)
        );

        local_diff = fabs(new_val - sharedMem[localY][localX]);
        matriz_new[globalY * N + globalX] = new_val;
    }

    __shared__ double block_diff[BLOCK_SIZE * BLOCK_SIZE];
    block_diff[threadIdx.y * BLOCK_SIZE + threadIdx.x] = local_diff;
    __syncthreads();

    // Redução para calcular a diferença média do bloco
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        double sum = 0.0;
        for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
            sum += block_diff[i];
        }
        atomicAdd_double(difmedio_device, sum);
    }
}


int main(){
    double *matriz_host_in;
    double *matriz_device_in, *matriz_device_new;

    //Tamanho em bytes da matriz
    size_t size = N * N * sizeof(double);

    //Aloca a matriz na memória do host
    matriz_host_in = (double*)malloc(size);

    if(matriz_host_in == NULL){
        printf("Erro ao alocar a matriz no host\n");
        exit(1);
    }

    // Alocar a matriz na memória do device
    cudaMalloc(&matriz_device_in, size);
    cudaMalloc(&matriz_device_new, size);

    if(matriz_device_in == NULL || matriz_device_new == NULL){
        printf("Erro ao alocar a matriz no device\n");
        exit(1);
    }

    // Configurar dimensões de bloco e grid
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Preechendo a matriz no host e preenchendo a matriz no device
    IniciarMatriz<<<gridSize, blockSize>>>(matriz_device_in);
    cudaDeviceSynchronize();

    // Copiar a matriz do device in para o device new
    cudaMemcpy(matriz_device_new, matriz_device_in, size, cudaMemcpyDeviceToDevice);
    // Copiar a matriz do device para o host
    cudaMemcpy(matriz_host_in, matriz_device_in, size, cudaMemcpyDeviceToHost);
    
    // Inicializar uma concentração alta no centro
    matriz_host_in[N/2 * N + N/2] = 1.0;

    // Copiar a matriz do host para o device
    cudaMemcpy(matriz_device_in, matriz_host_in, size, cudaMemcpyHostToDevice);

    // Executando as iterações no tempo para a equação de difusão
    double *difmedio_device;
    cudaMalloc(&difmedio_device, sizeof(double));

    for (int t = 0; t < T; t++) {
        cudaMemset(difmedio_device, 0, sizeof(double));
        diff_eq<<<gridSize, blockSize>>>(matriz_device_in, matriz_device_new, difmedio_device);
        cudaDeviceSynchronize();

        double difmedio_host;
        cudaMemcpy(&difmedio_host, difmedio_device, sizeof(double), cudaMemcpyDeviceToHost);
        difmedio_host /= ((N - 2) * (N - 2));

        if (t % 100 == 0) {
            printf("Iteração %d - Diferença média = %g\n", t, difmedio_host);
        }

        // Trocar matrizes
        double *temp = matriz_device_in;
        matriz_device_in = matriz_device_new;
        matriz_device_new = temp;
}
    cudaMemcpy(matriz_host_in, matriz_device_in, size, cudaMemcpyDeviceToHost);
    
    // Exibir resultado para verificação
    printf("Concentração final no centro: %f\n", matriz_host_in[N/2 * N + N/2]);
    
    // Liberar memória
    free(matriz_host_in);
    cudaFree(matriz_device_in);
    cudaFree(matriz_device_new);
    cudaFree(difmedio_device);

    return 0;
}