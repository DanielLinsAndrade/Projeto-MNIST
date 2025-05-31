from data_loader import carregar_dados_mnist

X_train, y_train, X_test, y_test = carregar_dados_mnist()

print("Imagens de treino:", X_train.shape)
print("Rótulos de treino:", y_train.shape)
print("Imagens de teste:", X_test.shape)
print("Rótulos de teste:", y_test.shape)
