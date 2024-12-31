# Simulação da Equação de Difusão com CUDA

Este projeto implementa a resolução da equação de difusão utilizando CUDA para paralelizar os cálculos em uma GPU. A simulação calcula a evolução de uma matriz de concentrações ao longo do tempo, considerando parâmetros ajustáveis como o coeficiente de difusão, número de iterações e tamanho da grade.

## Descrição das Principais Partes do Código

### 1. Definições de Parâmetros Globais
    ``` 
        #define N 2000       // Tamanho da grade
        #define T 1000       // Número de iterações
        #define D 0.1        // Coeficiente de difusão
        #define DELTA_T 0.01 // Passo de tempo
        #define DELTA_X 1.0  // Passo de espaço
        #define RADIUS 1     // Raio para cálculo
        #define BLOCK_SIZE 8 // Tamanho do bloco
    ```

Os parâmetros configuram a simulação, incluindo o tamanho da grade (N), número de iterações no tempo (T), coeficiente de difusão (D), e os tamanhos de passos de espaço e tempo (DELTA_X e DELTA_T). O BLOCK_SIZE define o tamanho de cada bloco para a execução CUDA.

### 2. Função IniciarMatriz
    ```	
        __global__ void IniciarMatriz(double *matriz)
    ```
    Essa função inicializa a matriz na memória do dispositivo com zeros. Cada thread calcula a posição (i, j) da matriz com base nos índices do bloco (blockIdx) e do thread (threadIdx).

### 3. Função atomicAdd_double

    ```
        __device__ double atomicAdd_double(double* address, double val)
    ```
    A função atomicAdd_double é uma implementação de atomicAdd para doubles, que não é nativamente suportada pelo CUDA. A função utiliza um lock para garantir a atomicidade da operação sem condições de corrida.

### 4. Função diff_eq

    ```
        __global__ void diff_eq(double *matriz, double *nova_matriz)
    ```
    A função diff_eq calcula a evolução da matriz de concentrações ao longo do tempo, utilizando a equação de difusão. Cada thread calcula a nova concentração de um ponto (i, j) da matriz com base nos pontos vizinhos e no coeficiente de difusão.

    Realiza o cálculo da equação de difusão.

    1. Uso de memória compartilhada: Armazena uma parte da matriz localmente em memória compartilhada para acesso mais rápido.
    2. Cálculo da equação de difusão: Utiliza os valores ao redor de um ponto para calcular a nova concentração com base no coeficiente de difusão D.
    3. Redução para calcular a diferença média: As diferenças locais são reduzidas dentro do bloco, e o valor final é somado à variável difmedio_device usando atomicAdd.