# Simulação da Equação de Difusão com MPI

Este projeto implementa a resolução da equação de difusão utilizando MPI para paralelizar os cálculos em múltiplos processos distribuídos. A simulação calcula a evolução de uma matriz de concentrações ao longo do tempo, considerando parâmetros ajustáveis como o coeficiente de difusão, número de iterações e tamanho da grade.

## Descrição das Principais Partes do Código

### 1. Definições de Parâmetros Globais
```c
#define N 16000 // Tamanho da matriz
#define T 1000  // Número de iterações
#define D 0.1   // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0
```
Os parâmetros configuram a simulação, incluindo o tamanho da matriz (N), o número de iterações no tempo (T), o coeficiente de difusão (D) e os tamanhos de passos de espaço e tempo (DELTA_X e DELTA_T). O número de processos influencia a distribuição das linhas da matriz entre os nós.

### 2. Função diff_eq
```c
void diff_eq(double **C, double **C_new, int local_rows, int rank, int size)
```
Essa função executa a evolução da matriz de concentração ao longo do tempo.

1. **Troca de bordas entre processos**: Cada processo MPI se comunica com seus vizinhos para trocar as linhas da borda, garantindo que os cálculos de difusão considerem corretamente os valores vizinhos.
2. **Cálculo da equação de difusão**: Cada ponto da matriz é atualizado com base nos seus vizinhos e no coeficiente de difusão.
3. **Cálculo da diferença média**: A diferença entre a matriz antiga e a nova é somada localmente e depois reduzida globalmente usando `MPI_Reduce`.

### 3. Função main
```c
int main(int argc, char** argv)
```
Principais etapas:
- **Inicialização do MPI**: O programa identifica o rank e o tamanho do comunicador MPI.
- **Alocação da matriz local**: Cada processo aloca apenas a parte da matriz que lhe cabe, mais duas linhas para bordas.
- **Distribuição inicial da concentração**: O processo central inicia com uma concentração maior no centro da matriz.
- **Laço de iterações**: A função `diff_eq` é chamada para calcular a equação de difusão.
- **Coleta dos resultados**: O `MPI_Gather` coleta os dados finais no processo mestre para análise.
- **Finalização do MPI**: Libera a memória alocada e finaliza o MPI.

## Resultados e Discussão

O código foi executado em uma máquina com as seguintes configurações:

- Processador: Intel Core i5-13420H
- Memória RAM: 8GB
- Sistema Operacional: Ubuntu 20.04 LTS
- MPI: OpenMPI
- Compilador: mpicc

O código foi compilado e executado com os comandos:

```bash
mpicc mpi.c -o mpi -lm
mpirun -np 4 ./mpi
```

Os resultados obtidos foram equivalentes à versão sequencial, indicando que a implementação paralela está correta. O tempo de execução reduziu significativamente conforme o número de processos aumentou, mostrando um bom aproveitamento da distribuição de carga.


## Versão MPI + OpenMP

Visando explorar ainda mais o paralelismo, uma versão híbrida com MPI e OpenMP foi implementada. O código foi modificado para distribuir as linhas da matriz entre os processos MPI, que por sua vez paralelizam o cálculo das iterações usando OpenMP.

### 1. Compilação
```bash
mpicc -fopenmp mpi_openmp.c -o mpi_openmp -lm
mpirun -np 4 ./mpi_openmp
```

### 2. Principais Modificações
- **Distribuição de Linhas**: Cada processo MPI recebe um número de linhas inteiras da matriz.
- **Paralelismo com OpenMP**: As iterações da equação de difusão são paralelizadas usando `#pragma omp parallel for`.

### 3. Resultados
A versão híbrida obteve um desempenho ainda melhor, com redução significativa no tempo de execução. A combinação de MPI e OpenMP permitiu explorar melhor o paralelismo disponível na máquina, resultando em uma execução mais rápida.
Essa versão não foi colocada no relatório final, mas está disponível no repositório do projeto.



