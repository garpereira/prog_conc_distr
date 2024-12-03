## Descrição do Projeto

Este projeto faz parte da disciplina de Programação Concorrente e Distribuída e tem como objetivo desenvolver uma aplicação que demonstre conceitos de concorrência e distribuição. A aplicação será dividida em várias partes, cada uma focando em diferentes aspectos da programação concorrente e distribuída, como sincronização de threads, comunicação entre processos e escalabilidade em sistemas distribuídos.



### Desafios Abordados:
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

- Configurar o ambiente, uma grade 2D onde cada célula representa a concentração de contaminantes em uma região do corpo d'água
- Definir os parâmetros do modelo de difusão, como o tamanho do domínio, o número de pontos de grade, o tempo de simulação, coeficiente de difusão, condições de contorno(por exemplo, bordas onde o contaminante não se espalha) e as condições iniciais (como uma área de alta concentraçãod e contaminante).

### Etapa 3: Implementação com OpenMP (Simulação Local em CPU)

- Implementar o modelo de difusão em uma CPU usando OpenMP para paralelizar o cálculo da concentração em cada célula da grade.
- Utilizar técnicas de sincronização e divisão de trabalho para garantir que múltiplas threads possam calcular a concentração de forma eficiente e sem condições de corrida.

#### Como rodar o código

1. **Compilar o código:**

   Para compilar o código com OpenMP, utilize o seguinte comando:
   ```bash
   gcc -fopenmp openmp.c -o openmp.o

2. **Executar o código:**

   Para executar o código compilado, utilize o comando:
   ```bash
   ./openmp.o
   
3. **Alterar o número de threads:**
   
   Caso queira alterar a quantidade de threads, é necessário modificar a linha no código:
   ```bash
   #define MAX_THREADS <Número de Threads>

  Substitua <Número de Threads> pelo valor desejado para o número de threads a ser utilizado na execução.

### Etapa 4: Implementação com CUDA (Simulação em GPU)

- Implementar o modelo de difusão em uma GPU usando CUDA para acelerar o cálculo da concentração em cada célula da grade.
- Utilizar blocos e threads CUDA para distribuir o trabalho entre os núcleos de GPU, aproveitando a paralelização em massa oferecida por esses dispositivos.

### Etapa 5: Implementação com MPI (Simulação Distribuída)

- Implementar o modelo de difusão em um cluster de computadores usando MPI para distribuir o cálculo da concentração em diferentes processos, podendo ser desenvolvido um código híbridro(incluindo trechso em OpenMP e CUDA).
- Utilizar comunicação ponto a ponto e coletiva para trocar informações entre os processos, permitindo que cada nó do cluster calcule uma parte da grade e compartilhe os resultados com os outros nós.


## Avaliação de Desempenho e Resultados



### Contribuição:
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.
