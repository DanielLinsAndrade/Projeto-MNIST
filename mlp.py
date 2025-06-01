import numpy as np


def relu(x):
    # A função ReLU recebe uma matriz ou vetor x, e para cada elemento que ela
    # recebe, retorna ele mesmo se for maior ou igual a 0, senão, retorna 0.
    return np.maximum(0, x)


def relu_derivada(x):
    # Essa versão do ReLU retorna 1 para elementos maiores que 0, o astype
    # transforma o resultado em um número decimal.
    return (x > 0).astype(float)


def softmax(x):
    # A função softmax transforma um vetor de números em probabilidades que
    # somam 1. Para evitar estouro numérico, subtraímos o maior valor de cada
    # linha (axis=1) da matriz. O keepdims=True mantém a forma original para a
    # operação funcionar direitinho. Depois calculamos a exponencial desses
    # valores ajustados e dividimos cada elemento pela soma da linha,
    # garantindo que a saída seja uma distribuição de probabilidade.
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(predicoes, rotulos):
    # A função de perda mede o quão ruim a previsão da rede está da resposta
    # correta, m é o número de exemplos, a quantidade de dados. O log_probs
    # é o logaritmo da matriz de probabilidades chamada de predições, onde
    # ela pega pra cada exemplo a probabilidade do acerto. O + 1e-9 evita
    # log de 0. O retorno é a soma de todos os valores de perda, o log_probs
    # dividida pelo número de exemplos, quanto menor o valor, melhor.
    m = rotulos.shape[0]
    log_probs = -np.log(predicoes[range(m), rotulos] + 1e-9)
    return np.sum(log_probs) / m


class MLP:
    # Essa é a rede neural em MPL! Iniciada com as entradas, neurônios,
    # classes, taxa de aprendizado
    def __init__(self, input_size=784, hidden_size=64, output_size=10,
                 taxa_aprendizado=0.01): 
        self.lr = taxa_aprendizado
        # Matrizes de pesos entre camadas, começam aleatórios
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        # bias
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Essa é a propragação para frente!
        # Resultado da multiplicação dos dados pelos pesoss + bias
        self.z1 = X @ self.w1 + self.b1
        # Aplicando ReLU
        self.a1 = relu(self.z1)
        # Entrada bruta da camada de saída
        self.z2 = self.a1 @ self.w2 + self.b2
        # Aplicando softmax pra transformar os a entrada em probabilidades
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y):
        # Propragação dos resultados para ajustar a rede!
        m = y.shape[0]
        y_one_hot = np.zeros((m, 10))
        y_one_hot[np.arange(m), y] = 1

        dz2 = self.a2 - y_one_hot
        dw2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2 @ self.w2.T * relu_derivada(self.z1)
        dw1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Atualiza pesos
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

    def treinar(self, X, y, epocas, batch_size=64):
        historico_perda = []
        for epoca in range(epocas):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                predicoes = self.forward(X_batch)
                self.backward(X_batch, y_batch)

            predicoes = self.forward(X)
            perda = cross_entropy(predicoes, y)
            historico_perda.append(perda)
            print(f"Época {epoca+1}/{epocas} - Perda: {perda:.4f}")
        return historico_perda
