# Parte principal do projeto responsável por:
# iniciar o treino, chamar os dados e criar as matrizes
#
# Explicação dos imports:
# avaliacao = importando do módulo de avaliação as funções estruturadas de acurácia e matriz.
# data_loader = importanto do módulo de carregar os dados a função de carregar os arquivos.
# MLP = importanto a classe MLP do módulo do Multilayer Perceptron.
# matplotlib.pyplot = importando biblioteca pra ajudar a plotar os dados gerados pela rede.
from avaliacao import calcular_acuracia, matriz_confusao, exibir_matriz_confusao
from data_loader import carregar_dados_mnist
from mlp import MLP

import matplotlib.pyplot as plt


# Carrega os dados dos arquivos do MNIST
X_train, y_train, X_test, y_test = carregar_dados_mnist()

# Inicializa a rede e passa os padrões de
# taxa de tamanho da camada oculta e taxa de aprendizado
# recebe o histórico de perdas conforme o passar das épocas
modelo = MLP(hidden_size=128, taxa_aprendizado=0.1)
historico = modelo.treinar(X_train, y_train, 10)
# o hidden_size=128 define o número de neurônios na camada oculta


# O hidden_size=128 define o número de neurônios na camada oculta
# A taxa_aprendizado=0.1 controla o tamanho dos passos durante o treinamento
# Faz o cálculo da precisão dos testes da
# rede com o passar das épocas com 4 casas de precisão
pred_test = modelo.forward(X_test)
acc = calcular_acuracia(pred_test, y_test)
print(f"Acurácia no teste: {acc:.4f}")

# Monta uma matriz de confusão 10x10
# comparando os valores reais dos valores preditos
# por fim, exibe a matriz de confusão.
y_predito = pred_test.argmax(axis=1)
matriz = matriz_confusao(y_test, y_predito)
exibir_matriz_confusao(matriz)
# a matriz de confusão é uma tabela que mostra a performance do modelo

# A matriz de confusão é uma ferramenta importante para análise de erros
# Estrutura o gráfico no matplot e exibe ele por fim
# exibe o gráfico baseado no histórico montado durante
# as épocas de treinamento da rede.
plt.plot(historico)
plt.title("Função de Perda por Época")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.grid(True)
plt.show()

# Exibe a acurácia final da rede

# este gráfico deve mostrar uma curva descendente, indicando que a perda
# diminui ao longo do treinamento.
