import matplotlib.pyplot as plt
import numpy as np

# Uma função que gera 25 imagens (ou quantas forem decididas) para gerar
# um comparativo entre as imagens do vetor de imagens reais e das imagens
# previstas pela rede, para demonstrar a precisão da rede, mostrando as imagens
# usando o matplot.
#
# X = são as imagens normalizadas
# y_real = são os rótulos reais.
# y_predito = são os rótulos que a rede acha que é real.
# quantidade = a quantidade de imagens que vão ser comparadas.
def mostrar_previsoes(X, y_real, y_predito, quantidade=25):
    assert len(X) == len(y_real) == len(y_predito)
    indices = np.random.choice(len(X), size=quantidade, replace=False)

    # Tamanho das figuras exibidas pelo matplot.
    plt.figure(figsize=(12, 12))
    lado = int(np.ceil(np.sqrt(quantidade)))

    # Um laço que vai passar comparando a precisão dos números entre
    # os vetores e vai atribuir uma cor se tiver certo ou errado, caso
    # esteja certo (o número é igual no MNIST e a rede previu certo): Verde,
    # caso esteja errado: Vermelho.
    for i, idx in enumerate(indices):
        imagem = X[idx].reshape(28, 28)
        real = y_real[idx]
        previsto = y_predito[idx]
        cor = "green" if real == previsto else "red"

        plt.subplot(lado, lado, i+1)
        plt.imshow(imagem, cmap="gray")
        plt.title(f"Real: {real} | Previsto: {previsto}", color=cor)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
