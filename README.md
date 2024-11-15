# BackPropagation

## Descrição

Este projeto implementa uma rede neural simples utilizando o algoritmo de **Backpropagation** para classificar a cor predominante a partir de entradas RGB. A rede foi treinada para reconhecer as cores principais `Vermelho`, `Verde` e `Azul`, utilizando valores normalizados de RGB como entradas. Com base nos dados de entrada, a rede retorna a cor correspondente, fornecendo uma classificação precisa das cores a partir de seus valores RGB. 

O sistema também oferece uma funcionalidade para capturar e processar dados de cores em tempo real utilizando a câmera do dispositivo.

## Funcionalidades

- **Classificação de cores RGB**:
  - Identifica as cores predominantes (`Vermelho`, `Verde`, `Azul`) com base nos valores RGB normalizados.
- **Treinamento da rede neural**:
  - Treinamento pode ser feito com ou sem o uso de câmera.
- **Modo de captura ao vivo**:
  - Processa e classifica cores em tempo real utilizando a câmera integrada.
- **Acompanhamento do treinamento**:
  - Exibe o erro médio a cada 1000 épocas durante o treinamento sem câmera.

## Estrutura

- **Funções de ativação**:
  - **`sigmoid(x)`**: Função de ativação que normaliza os valores.
  - **`sigmoid_derivative(x)`**: Derivada da função sigmoid usada no processo de backpropagation para ajuste de pesos.
  
- **Configuração da rede neural**:
  - **Camada de entrada**: 3 neurônios representando os valores RGB da entrada.
  - **Camada oculta**: 4 neurônios para o processamento intermediário.
  - **Camada de saída**: 3 neurônios correspondentes às cores `Vermelho`, `Verde` e `Azul`.

- **Treinamento sem câmera**:
  - Baseado em um conjunto fixo de valores RGB.
- **Treinamento com câmera**:
  - Processa imagens capturadas ao vivo, extraindo a cor média de uma região central para treino e classificação.

## Pré-requisitos

Certifique-se de ter o Python instalado e os pacotes necessários configurados. Instale os pacotes `NumPy` e `OpenCV` com o seguinte comando:

```Powershell
pip install numpy opencv-python
