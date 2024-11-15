import numpy as np
import cv2

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
input_layer_neurons = 3
hidden_layer_neurons = 4
output_neurons = 3
learning_rate = 0.1
max_error_threshold = 0.01

# Inicializando os pesos
np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

def color_description(output):
    color_index = np.argmax(output)
    return color_labels[color_index]

def train_with_camera():
    global weights_input_hidden, weights_hidden_output
    cap = cv2.VideoCapture(0)
    print("Aponte para uma cor (Vermelho, Verde ou Azul). Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a câmera.")
            break

        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        roi_size = 50
        roi = frame[center_y - roi_size:center_y + roi_size, center_x - roi_size:center_x + roi_size]
        avg_color = roi.mean(axis=(0, 1)) / 255.0
        r, g, b = avg_color
        X = np.array([[r, g, b]])

        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_activation = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)
        confidence = np.max(predicted_output)

        if confidence >= 0.7:
            detected_color = color_description(predicted_output)
            print(f"Cor detectada: {detected_color} com confiança {confidence:.2f}")
        else:
            print("Nenhuma cor detectada.")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train_without_camera():
    global weights_input_hidden, weights_hidden_output  # Declarar as variáveis como globais

    X = np.array([
        [0.9, 0.1, 0.1],  # Vermelho
        [0.1, 0.9, 0.1],  # Verde
        [0.1, 0.1, 0.9],  # Azul
        [0.9, 0.1, 0.1],  # Vermelho (repetido para balancear as dimensões)
        [0.1, 0.9, 0.1],  # Verde (repetido)
        [0.1, 0.1, 0.9],  # Azul (repetido)
    ])

    # Expandindo `y` para ter a mesma forma de `X`
    y_expanded = np.tile(y, (2, 1))  # Repete `y` duas vezes ao longo do eixo 0

    epoch = 0
    while True:
        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_activation = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)
        error = y_expanded - predicted_output  # Ajustado para usar `y_expanded`

        if epoch % 1000 == 0:
            print(f"Erro na época {epoch}: {np.mean(np.abs(error))}")

        if np.mean(np.abs(error)) <= max_error_threshold:
            print("Erro abaixo do limite. Processo concluído.")
            break

        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)
        weights_hidden_output += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
        epoch += 1

def main():
    while True:
        print("\nEscolha uma opção:")
        print("1. Com Câmera")
        print("2. Sem Câmera")
        print("3. Sair")
        choice = input("Digite sua escolha: ")

        if choice == "1":
            train_with_camera()
        elif choice == "2":
            train_without_camera()
        elif choice == "3":
            print("Encerrando o programa.")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
