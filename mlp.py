import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivada(x):
    return (x > 0).astype(float)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(predicoes, rotulos):
    m = rotulos.shape[0]
    log_probs = -np.log(predicoes[range(m), rotulos] + 1e-9)
    return np.sum(log_probs) / m

class MLP:
    def __init__(self, input_size=784, hidden_size=64, output_size=10, taxa_aprendizado=0.01):
        self.lr = taxa_aprendizado
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.w1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y):
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
