# Simulação de Difusão de Contaminantes com Computação Paralela e Distribuída

## Descrição do Projeto

Este projeto faz parte da disciplina de Programação Concorrente e Distribuída e tem como objetivo desenvolver uma aplicação que demonstre conceitos de concorrência e distribuição. A aplicação será dividida em várias partes, cada uma focando em diferentes aspectos da programação concorrente e distribuída, como sincronização de threads, comunicação entre processos e escalabilidade em sistemas distribuídos.

## Desafios Abordados:
1. **Sincronização de Threads**: Garantir que múltiplas threads possam acessar recursos compartilhados sem causar condições de corrida ou inconsistências nos dados.
2. **Comunicação entre Processos**: Facilitar a troca de informações entre processos que podem estar executando em diferentes máquinas ou núcleos de processamento.
3. **Escalabilidade**: Desenvolver soluções que possam crescer em capacidade e desempenho à medida que mais recursos computacionais são adicionados.

## Abordagem

Para enfrentar esses desafios, o projeto utiliza uma combinação de métodos e tecnologias avançadas:

- **MPI (Message Passing Interface)**: Utilizado para comunicação eficiente entre processos em sistemas distribuídos. MPI permite a troca de mensagens entre processos que podem estar em diferentes nós de um cluster, facilitando a paralelização de tarefas.
- **OpenMP (Open Multi-Processing)**: Uma API que suporta programação paralela em plataformas multiprocessadas. OpenMP é utilizado para paralelizar loops e seções de código, permitindo que múltiplas threads executem tarefas simultaneamente.
- **Pthreads (POSIX Threads)**: Uma biblioteca padrão para programação com threads em sistemas Unix. Pthreads é utilizado para criar e gerenciar threads, oferecendo controle fino sobre a sincronização e comunicação entre elas.
- **CUDA (Compute Unified Device Architecture)**: Uma plataforma de computação paralela e API da NVIDIA que permite o uso de GPUs para processamento geral. CUDA é utilizado para acelerar tarefas computacionalmente intensivas, distribuindo o trabalho entre milhares de núcleos de GPU.

### Benefícios da Abordagem:
- **Eficiência**: A utilização de threads e processos permite que tarefas sejam executadas em paralelo, reduzindo o tempo total de execução.
- **Escalabilidade**: As soluções desenvolvidas podem ser escaladas para aproveitar recursos adicionais, como mais núcleos de CPU ou GPUs.
- **Flexibilidade**: A combinação de diferentes tecnologias permite abordar uma ampla gama de problemas, desde a sincronização de threads até a comunicação entre processos distribuídos.

Essa abordagem integrada permite que o projeto explore de forma abrangente os conceitos de programação concorrente e distribuída, oferecendo soluções práticas para problemas reais de processamento de dados em larga escala.

## Etapas do Projeto

### Etapa 1: Modelo de Difusão

Estudar a Equação de Difusão/Transporte, que é representada por:

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20%3D%20D%20%5Cnabla%20%5E2%20u)

Onde:
- ![equation](https://latex.codecogs.com/gif.latex?u) é a concentração de um material em um ponto do espaço.
- ![equation](https://latex.codecogs.com/gif.latex?D) é o coeficiente de difusão.
- ![equation](https://latex.codecogs.com/gif.latex?%5Cnabla%20%5E2) é o operador Laplaciano, ou seja, a taxa de variação espacial da concentração.
- ![equation](https://latex.codecogs.com/gif.latex?t) é o tempo.

### Etapa 2: Configuração do Ambiente e Parametrização

- Configurar o ambiente com uma grade 2D onde cada célula representa a concentração de contaminantes em uma região do corpo d'água.
- Definir os parâmetros do modelo de difusão, como:
   - **Tamanho do domínio:** `N x N`
   - **Número de pontos de grade:** `#define N 2000`
   - **Número de iterações:** `#define T 1000`
   - **Coeficiente de difusão:** `#define D 0.1`
   - **Passos de tempo e espaço:** `#define DELTA_T 0.01`, `#define DELTA_X 1.0`
   - **Condições de contorno:** Bordas onde o contaminante não se espalha.
   - **Condições iniciais:** Área central com alta concentração de contaminantes.

### Etapa 3: Implementação com OpenMP (Simulação Local em CPU)

A implementação com OpenMP paraleliza os loops da simulação utilizando múltiplos núcleos da CPU. As diretivas utilizadas incluem:

- **`#pragma omp parallel for`**: Para dividir os cálculos entre múltiplos threads.
- **`#pragma omp reduction`**: Para somar os valores de erro médio entre threads de forma segura.
- **`#pragma omp collapse(2)`**: Para paralelizar duplos loops aninhados.

#### Como rodar o código

1. **Compilar o código com OpenMP:**
    ```bash
    cd openmp
    gcc -fopenmp openmp.c -o openmp.o
    ```
2. **Executar o código:**
    ```bash
    ./openmp.o
    ```
3. **Alterar o número de threads: Modifique a linha no código:**
    ```c
    #define MAX_THREADS <Número de Threads>
    ```

### Etapa 4: Implementação com CUDA (Simulação em GPU)

A versão CUDA paraleliza os cálculos distribuindo-os entre milhares de núcleos da GPU. A implementação utiliza:

- **Memória compartilhada (`__shared__`)**: Para otimizar acessos e reduzir latência.
- **Kernel CUDA**: Para distribuir os cálculos entre blocos e threads.
- **`atomicAdd`**: Para garantir a soma segura das diferenças no cálculo do erro médio.

#### Como rodar o código

1. **Compilar o código com CUDA:**
    ```bash
    cd cuda
    nvcc cuda.cu -o cuda.o
    ```
2. **Executar o código:**
    ```bash
    ./cuda.o
    ```
3. **Alterar o número de threads e blocos: Modifique as linhas no código:**
    ```c
    #define THREADS_PER_BLOCK <Número de Threads por Bloco>
    #define BLOCKS <Número de Blocos>
    ```

Essa implementação permite que a simulação aproveite o poder de processamento paralelo das GPUs, acelerando significativamente o tempo de execução.

### Etapa 5: Implementação com MPI (Simulação Distribuída)

A versão MPI divide a matriz entre múltiplos processos, utilizando:

- **MPI_Sendrecv**: Para troca de informações entre processos vizinhos.
- **MPI_Reduce**: Para somar os erros médios de cada processo.
- **MPI_Gather**: Para coletar os resultados no processo mestre.

#### Como rodar o código

1. **Compilar o código com MPI:**
    ```bash
    cd mpi
    mpicc mpi.c -o mpi.o
    ```
2. **Executar o código:**
    ```bash
    mpirun -np <Número de Processos> ./mpi.o
    ```

Essa implementação permite que a simulação seja distribuída entre múltiplos nós de um cluster, aproveitando a capacidade de processamento paralelo de sistemas distribuídos.

## Avaliação de Desempenho e Resultados

Os testes de desempenho foram realizados comparando OpenMP, CUDA e MPI.

| Tamanho da Grade | OpenMP (s) | CUDA (s) | MPI (s, P=4) |
|------------------|------------|----------|--------------|
| 2000 x 2000      | 11.7326    | 4.7988   | 20.644       |
| 4000 x 4000      | 43.5072    | 13.7806  | 41.979       |
| 8000 x 8000      | 129.8362   | 46.951   | 139.416      |

Os resultados mostram que:

- CUDA foi a abordagem mais rápida para grades grandes.
- MPI teve desempenho variável dependendo do número de processos.
- OpenMP teve melhor eficiência para grades menores.