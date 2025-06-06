# Função de Avaliação do Treinamento
import numpy as np


def calcular_acuracia(predicoes, rotulos):
    # Essa função calcula a acurácia da rede comparando com os rótulos
    # verdadeiros: O np.argmax transforma a saída do softmax no número da
    # classe predita e põe no pred_labels. Ex: Entre [0.1 , 0.7 e 0.5], ele
    # vai retornar 1, a posição da maior porcentagem. O axis indica onde
    # estamos procurando. Ex: 0 = por coluna, 1 = por linha. O retorno é a
    # média predição comparada aos rótulos, onde o "==" retorna 1 para
    # correspondência e 0 para divergência. Aí o np.mean faz a média dos
    # acertos pelo total.
    pred_labels = np.argmax(predicoes, axis=1)
    return np.mean(pred_labels == rotulos)


def matriz_confusao(rotulos_verdadeiros, rotulos_preditos, num_classes=10):
    # Essa função cria uma tabela que mostra onde o modelo acertou e errou,
    # onde cada linha é o que era pra ser e cada coluna é a resposta do modelo
    # Primeiro se cria a matriz zerada, e o for vai percorrer os pares para
    # gerar a dupla (esperado, real). Onde o valor da comparação for igual,
    # vai ser 1. Exemplo:
    # Se verdadeiro = 1 e predito = 1 → matriz[1][1] += 1 → acerto.
    # Se verdadeiro = 0 e predito = 1 → matriz[0][1] += 1 → erro.
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for verdadeiro, predito in zip(rotulos_verdadeiros, rotulos_preditos):
        matriz[verdadeiro, predito] += 1
    return matriz
    #a matriz de confusão é útil para indentificar padrões de erros


def exibir_matriz_confusao(matriz):
    # Essa função imprime a matriz de confusão construída com a função
    # matriz_confusao.
    print("Matriz de Confusão:")
    print("    " + " ".join(f"{i:^5}" for i in range(matriz.shape[1])))
    for i, linha in enumerate(matriz):
        print(f"{i}: " + " ".join(f"{val:^5}" for val in linha))
