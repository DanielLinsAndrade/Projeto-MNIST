from mlp import MLP
from data_loader import carregar_dados_mnist
from avaliacao import calcular_acuracia, matriz_confusao, exibir_matriz_confusao
import matplotlib.pyplot as plt

# Carrega os dados
X_train, y_train, X_test, y_test = carregar_dados_mnist()

# Instancia e treina a rede
modelo = MLP(hidden_size=128, taxa_aprendizado=0.1)
historico = modelo.treinar(X_train, y_train, 5)

# Avaliação
pred_test = modelo.forward(X_test)
acc = calcular_acuracia(pred_test, y_test)
print(f"Acurácia no teste: {acc:.4f}")

# Matriz de confusão
y_predito = pred_test.argmax(axis=1)
matriz = matriz_confusao(y_test, y_predito)
exibir_matriz_confusao(matriz)

# Gráfico da função de perda
plt.plot(historico)
plt.title("Função de Perda por Época")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.grid(True)
plt.show()
