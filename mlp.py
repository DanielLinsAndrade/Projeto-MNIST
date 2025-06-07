import numpy as np
from ativacoes import relu, relu_derivada, softmax
from perdas import cross_entropy


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
        # Multiplicação por 0.01 nos pesos iniciais é uma técnica importante
        # para evitar que os valores fiquem muito grandes no início do treinamento,
        # o que poderia causar saturação dos neurônios e dificultar o aprendizado.

    def forward(self, X):
        # Essa é a propragação para frente!
        # Resultado da multiplicação dos dados pelos pesoss + bias
        self.z1 = X @ self.w1 + self.b1 # O operador @ é uma multiplicação de matrizes em Python (numpy)
        # Aplicando ReLU
        self.a1 = relu(self.z1)
        # Entrada bruta da camada de saída
        self.z2 = self.a1 @ self.w2 + self.b2
        # Aplicando softmax pra transformar os a entrada em probabilidades
        self.a2 = softmax(self.z2)
        return self.a2


    def backward(self, X, y):
        # Propragação dos resultados para ajustar a rede!
        m = y.shape[0]  # Esse é o número de exemplos dentro do batch
        # Isso é como a rede vai entender qual é o número, onde ele vai
        # correponder a posição dentro do vetor. Primeiro cria a matriz
        # cheia de zeros com m(num exemplos) e 10 colunas (qtd classes)
        y_one_hot = np.zeros((m, 10))
        # e depois essa linha coloca 1 na coluna correta da classe de cada
        # exemplo (3 = 0001000000)
        y_one_hot[np.arange(m), y] = 1
        # Esse é o erro da saída, previsão - verdade
        dz2 = self.a2 - y_one_hot
        # Esses são os gradientes para peso e bias da segunda camada
        dw2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        # Calculo do erro de cada neurônio da camada oculta
        dz1 = dz2 @ self.w2.T * relu_derivada(self.z1)
        # Variação dos pesos da camada oculta
        dw1 = X.T @ dz1 / m
        # Calculo do gradiente do bias da camada oculta
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Atualiza pesos, subtraindo o gradiente vezes a taxa de aprendizado,
        # melhorando as previsões
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

        # processo conhecido como Gradient Descent, onde se ajusta os parâmetros
        # na direção oposta ao gradiente para minimizar a função de perda.

    def treinar(self, X, y, epocas, batch_size=64):
        # Ela recebe os dados, rótulos, num epocas e tam batch
        historico_perda = []  # Guarda a perda de cada época
        for epoca in range(epocas):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)  # embaralhando os dados
            X, y = X[indices], y[indices]  # Reorganizando dados e rótulos

            # o embaralhamento dos dados é crucial para evitar viés no treinamento
            # e ajuda a rede a generalizar melhor, evitando que ela memorize a ordem dos exemplos

            # Treinamento em conjunto de dados (batches)
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                predicoes = self.forward(X_batch)  # passo foward
                self.backward(X_batch, y_batch)  # passo backward

            # Calculo da perda total após cada época ser analisada
            predicoes = self.forward(X)
            perda = cross_entropy(predicoes, y)
            historico_perda.append(perda)
            print(f"Época {epoca+1}/{epocas} - Perda: {perda:.8f}")
        return historico_perda
