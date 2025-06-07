import numpy as np

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