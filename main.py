# Parte principal do projeto responsável por:
# iniciar o treino, chamar os dados e criar as matrizes
#
# Explicação dos imports:
# avaliacao = importando do módulo de avaliação as funções estruturadas de acurácia e matriz.
# data_loader = importanto do módulo de carregar os dados a função de carregar os arquivos.
# MLP = importanto a classe MLP do módulo do Multilayer Perceptron.
# matplotlib.pyplot = importando biblioteca pra ajudar a plotar os dados gerados pela rede.
# visualizacao = importando o módulo onde tem a função para comparar as imagens
# previstas com as não previstas.
from avaliacao import calcular_acuracia, matriz_confusao, exibir_matriz_confusao
from data_loader import carregar_dados_mnist
from visualizacao import mostrar_previsoes
from mlp import MLP

import matplotlib.pyplot as plt


# Carrega os dados dos arquivos do MNIST
X_train, y_train, X_test, y_test = carregar_dados_mnist()

# Inicializa a rede e passa os padrões de
# taxa de tamanho da camada oculta e taxa de aprendizado
# recebe o histórico de perdas conforme o passar das épocas
modelo = MLP(hidden_size=128, taxa_aprendizado=0.1)
historico = modelo.treinar(X_train, y_train, 10)

# Faz o cálculo da precisão dos testes da
# rede com o passar das épocas com 4 casas de precisão
pred_test = modelo.forward(X_test)
acc = calcular_acuracia(pred_test, y_test)
print(f"Acurácia no teste: {acc:.8f}")

# Monta uma matriz de confusão 10x10
# comparando os valores reais dos valores preditos
# por fim, exibe a matriz de confusão.
y_predito = pred_test.argmax(axis=1)
matriz = matriz_confusao(y_test, y_predito)
exibir_matriz_confusao(matriz)

# Após ter as variáveis de rótulo prontas, chama uma função
# para visualizar a previsão da rede mostrando os números.
mostrar_previsoes(X_test, y_test, y_predito, quantidade=25)

# Estrutura o gráfico no matplot e exibe ele por fim
# exibe o gráfico baseado no histórico montado durante
# as épocas de treinamento da rede.
plt.plot(historico)
plt.title("Função de Perda por Época")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.grid(True)
plt.show()
