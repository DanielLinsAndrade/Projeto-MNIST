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
