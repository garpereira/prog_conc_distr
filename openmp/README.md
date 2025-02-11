# Simulação da Equação de Difusão com OpenMP

Este projeto implementa a resolução da equação de difusão utilizando OpenMP para paralelizar os cálculos em uma CPU. A simulação calcula a evolução de uma matriz de concentrações ao longo do tempo, considerando parâmetros ajustáveis como o coeficiente de difusão, número de iterações e tamanho da grade.

## Descrição das Principais Partes do Código

### 1. Definições de Parâmetros Globais
```c
#define N 2000       // Tamanho da grade
#define T 1000       // Número de iterações
#define D 0.1        // Coeficiente de difusão
#define DELTA_T 0.01 // Passo de tempo
#define DELTA_X 1.0  // Passo de espaço
#define MAX_THREADS 16 // Número máximo de threads
```

Os parâmetros configuram a simulação, incluindo o tamanho da grade (N), número de iterações no tempo (T), coeficiente de difusão (D), e os tamanhos de passos de espaço e tempo (DELTA_X e DELTA_T). O MAX_THREADS define o número máximo de threads para a execução OpenMP.

### 2. Função diff_eq
```c
void diff_eq(double **C, double **C_new)
```
A função `diff_eq` calcula a evolução da matriz de concentrações ao longo do tempo utilizando OpenMP para paralelizar os cálculos.

1. **Cálculo da equação de difusão**: Cada thread calcula a nova concentração de um ponto (i, j) da matriz com base nos pontos vizinhos e no coeficiente de difusão.
2. **Uso de `#pragma omp parallel for`**: Paraleliza o loop para atualização da matriz e para a soma da diferença média entre iterações.
3. **Monitoramento da diferença média**: A cada 100 iterações, a diferença média é impressa para acompanhar a convergência da simulação.

### 3. Função main
```c
int main()
```
Principais etapas:
- **Alocação de memória**: A matriz é alocada dinamicamente para armazenar as concentrações.
- **Inicialização da matriz**: A matriz é preenchida com zeros, exceto por um valor inicial alto no centro, que simula uma concentração inicial.
- **Execução da equação de difusão**: A função `diff_eq` é chamada para processar a evolução da matriz ao longo das iterações.
- **Resultados e limpeza**: O programa exibe a concentração final no centro da matriz.

## Resultados e Discussão

O código foi executado em uma máquina com as seguintes configurações:

- Processador: Intel Core i5-13420H
- Memória RAM: 8GB
- Sistema Operacional: Ubuntu 20.04 LTS
- OpenMP ativado
- Compilador: gcc

O código foi compilado e executado com os comandos:

```bash
gcc -fopenmp openmp.c -o openmp -lm
./openmp
```

Os resultados foram equivalentes à versão sequencial, demonstrando que a implementação paralela está correta. A aceleração obtida com OpenMP foi significativa, com uma redução no tempo de execução proporcional ao número de threads utilizadas.