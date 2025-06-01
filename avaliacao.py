import numpy as np

def calcular_acuracia(predicoes, rotulos):
    pred_labels = np.argmax(predicoes, axis=1)
    return np.mean(pred_labels == rotulos)

def matriz_confusao(rotulos_verdadeiros, rotulos_preditos, num_classes=10):
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for verdadeiro, predito in zip(rotulos_verdadeiros, rotulos_preditos):
        matriz[verdadeiro, predito] += 1
    return matriz

def exibir_matriz_confusao(matriz):
    print("Matriz de Confusão:")
    print("    " + " ".join(f"{i:^5}" for i in range(matriz.shape[1])))
    for i, linha in enumerate(matriz):
        print(f"{i}: " + " ".join(f"{val:^5}" for val in linha))
