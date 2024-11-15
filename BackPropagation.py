import numpy as np
import cv2
import time  # Importar o módulo time

# Funções de ativação sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Dados estáticos de saída esperada (cores para treinamento)
color_labels = ["Vermelho", "Verde", "Azul"]
y = np.array([
    [1, 0, 0],  # Vermelho
    [0, 1, 0],  # Verde
    [0, 0, 1]   # Azul
])

# Configuração da rede neural
input_layer_neurons = 3          # Neurônios de entrada (R, G, B)
hidden_layer_neurons = 4         # Neurônios na camada oculta
output_neurons = 3               # Neurônios de saída (cores)
learning_rate = 0.1
epochs = 10000
max_error_threshold = 0.01

# Inicialização dos pesos - fora de qualquer função para garantir que sejam globais
np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

# Função para mapear a saída prevista para a descrição da cor
def color_description(output):
    color_index = np.argmax(output)  # Índice da cor com maior valor
    return color_labels[color_index]

# Função para capturar dados da câmera e treinar em tempo real
def train_with_camera():
    global weights_input_hidden, weights_hidden_output  # Garante que estamos usando as variáveis globais
    cap = cv2.VideoCapture(0)
    print("Aponte para uma cor (Vermelho, Verde ou Azul).")
    print("Pressione 'q' para sair do treinamento.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calcula o centro da imagem e define a ROI
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        roi_size = 50
        roi = frame[center_y - roi_size:center_y + roi_size, center_x - roi_size:center_x + roi_size]
        avg_color = roi.mean(axis=(0, 1)) / 255.0  # Normaliza valores RGB entre 0 e 1

        # Entrada atual (valores RGB normalizados)
        r, g, b = avg_color
        X = np.array([[r, g, b]])

        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_activation = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)

        # Verifica a confiança da predição (limiar de 0.7)
        confidence_threshold = 0.7
        confidence = np.max(predicted_output)

        # Se a confiança for alta, exibe a cor detectada, caso contrário, exibe "Nenhuma Cor"
        if confidence >= confidence_threshold:
            detected_color = color_description(predicted_output)
            print(f"Cor detectada: {detected_color} com confiança {confidence:.2f}")
        else:
            print("Nenhuma Cor detectada.")

        # Espera por entrada do usuário para adicionar a cor ao treinamento
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            target = y[0]  # Vermelho
        elif key == ord('g'):
            target = y[1]  # Verde
        elif key == ord('b'):
            target = y[2]  # Azul
        elif key == ord('q'):
            print("Encerrando treinamento.")
            break
        else:
            continue  # Aguarda uma entrada válida

        # Calcula o erro e atualiza os pesos
        error = target - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)
        weights_hidden_output += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

        # Mostra a captura da câmera com a ROI
        cv2.rectangle(frame, (center_x - roi_size, center_y - roi_size),
                      (center_x + roi_size, center_y + roi_size), (0, 255, 0), 2)
        cv2.imshow("Treinamento com Câmera", frame)

    cap.release()
    cv2.destroyAllWindows()

# Executa o treinamento com dados da câmera
train_with_camera()
