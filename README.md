# BackPropagation

## Descrição

Este projeto implementa uma rede neural simples utilizando o algoritmo de **Backpropagation** para classificar a cor predominante a partir de entradas RGB. A rede foi treinada para reconhecer as cores principais `Vermelho`, `Verde` e `Azul`, utilizando valores normalizados de RGB como entradas. Com base nos dados de entrada, a rede retorna a cor correspondente, fornecendo uma classificação precisa das cores a partir de seus valores RGB.

## Estrutura

- **Funções de Ativação**:
  - **`sigmoid(x)`**: Função de ativação para normalizar os valores.
  - **`sigmoid_derivative(x)`**: Derivada da função sigmoid usada no processo de backpropagation para atualizar os pesos da rede.
  
- **Camadas da Rede Neural**:
  - **Camada de Entrada**: Três neurônios que representam os valores RGB da entrada.
  - **Camada Oculta**: Quatro neurônios para realizar o processamento intermediário entre a entrada e a saída.
  - **Camada de Saída**: Três neurônios correspondentes às cores `Vermelho`, `Verde` e `Azul`.

- **Treinamento**:
  - A rede é treinada utilizando um conjunto de dados com valores RGB e saídas codificadas em one-hot para as cores. 
  - O erro da rede é calculado e exibido a cada 1000 épocas de treinamento para acompanhar o progresso da aprendizagem.

## Pré-requisitos

Certifique-se de ter o Python instalado e o pacote `NumPy` para executar o código. Você pode instalá-lo executando o comando abaixo no terminal ou PowerShell:

Execute o código Python:
    ```
    python BackPropagation.py
    ```

Instalação do Numpy:
`pip install NumPy`

O código exibirá a classificação das cores com base nos valores RGB fornecidos.

## Como Funciona

A função `detect_color(r, g, b)` recebe os valores RGB normalizados (entre 0 e 1) como entrada, realiza o cálculo do feedforward através das camadas da rede, e retorna a cor classificada. 

Exemplo de uso:

```python
print(detect_color(0.9, 0.2, 0.1))  # Esperado: Vermelho
print(detect_color(0.2, 0.8, 0.2))  # Esperado: Verde
print(detect_color(0.1, 0.1, 0.9))  # Esperado: Azul
