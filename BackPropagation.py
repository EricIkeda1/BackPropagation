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

    # Verificando se a câmera foi aberta corretamente
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return

    # Carregar o classificador em cascata Haar para detecção de rosto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("Aponte para uma cor (Vermelho, Verde ou Azul). Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a câmera.")
            break

        # Convertendo a imagem para o espaço de cor HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Definindo os limites das cores no espaço HSV (ajustados para cada cor)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        lower_blue = np.array([100, 150, 50])  # Ajustado para o azul
        upper_blue = np.array([140, 255, 255])  # Ajustado para o azul

        # Criando máscaras para cada cor
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Criando uma imagem para mostrar todas as máscaras juntas
        combined_mask = cv2.merge([mask_red, mask_green, mask_blue])

        # Exibe a imagem combinada das máscaras
        cv2.imshow("Máscaras Combinadas", combined_mask)

        # Verificando a predominância de cada cor
        red_area = np.sum(mask_red) / (frame.shape[0] * frame.shape[1])
        green_area = np.sum(mask_green) / (frame.shape[0] * frame.shape[1])
        blue_area = np.sum(mask_blue) / (frame.shape[0] * frame.shape[1])

        # Determinando qual cor é a predominante
        if red_area > 0.1:
            detected_color = "Vermelho"
            confidence = red_area
        elif green_area > 0.1:
            detected_color = "Verde"
            confidence = green_area
        elif blue_area > 0.1:
            detected_color = "Azul"
            confidence = blue_area
        else:
            detected_color = "Nenhuma cor predominante"
            confidence = 0

        if confidence >= 0.1:
            print(f"Cor detectada: {detected_color} com confiança {confidence:.2f}")
        else:
            print("Nenhuma cor detectada com alta confiança.")

        # Convertendo para escala de cinza (necessário para a detecção de rostos)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectando rostos na imagem
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Desenhando um retângulo ao redor do rosto detectado
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Cor azul para o retângulo

        # Exibe a imagem original da câmera com o retângulo ao redor do rosto
        cv2.imshow("Imagem com Detecção de Rosto", frame)

        # Verifica se a tecla 'q' foi pressionada para sair
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train_without_camera():
    global weights_input_hidden, weights_hidden_output

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
        print("1. Treinamento com Câmera")
        print("2. Treinamento sem Câmera")
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
