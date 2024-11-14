import numpy as np

# Função de ativação sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Dados de entrada (média RGB de diferentes cores) e saídas (classificação)
# X -> [R, G, B], y -> [1, 0, 0] para vermelho, [0, 1, 0] para verde, [0, 0, 1] para azul
X = np.array([
    [0.9, 0.1, 0.1],  # Vermelho predominante
    [0.1, 0.9, 0.1],  # Verde predominante
    [0.1, 0.1, 0.9],  # Azul predominante
    [0.5, 0.2, 0.2],  # Vermelho médio
    [0.2, 0.5, 0.3],  # Verde médio
    [0.3, 0.3, 0.7],  # Azul médio
])

# Saída esperada (codificação one-hot)
y = np.array([
    [1, 0, 0],  # Vermelho
    [0, 1, 0],  # Verde
    [0, 0, 1],  # Azul
    [1, 0, 0],  # Vermelho
    [0, 1, 0],  # Verde
    [0, 0, 1],  # Azul
])

# Definindo os hiperparâmetros da rede
input_layer_neurons = X.shape[1]  # Número de neurônios na camada de entrada (R, G, B)
hidden_layer_neurons = 4          # Número de neurônios na camada oculta
output_neurons = 3                # Número de neurônios na camada de saída (vermelho, verde, azul)
learning_rate = 0.1               # Taxa de aprendizado
epochs = 10000                    # Número de épocas de treinamento

# Inicializando os pesos aleatoriamente
np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

# Treinamento da rede
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    # Cálculo do erro
    error = y - predicted_output
    if epoch % 1000 == 0:
        print(f"Erro na época {epoch}: {np.mean(np.abs(error))}")

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

    # Atualização dos pesos
    weights_hidden_output += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

# Função para mapear a saída prevista para a descrição da cor
def color_description(output):
    color_labels = ["Vermelho", "Verde", "Azul"]
    color_index = np.argmax(output)  # Encontra o índice da cor com maior valor
    return color_labels[color_index]

# Função para detectar a cor com base em valores RGB
def detect_color(r, g, b):
    input_data = np.array([[r, g, b]])
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)
    return color_description(predicted_output)

# Testando com novas cores
print(detect_color(0.9, 0.2, 0.1))  # Esperado: Vermelho
print(detect_color(0.2, 0.8, 0.2))  # Esperado: Verde
print(detect_color(0.1, 0.1, 0.9))  # Esperado: Azul
