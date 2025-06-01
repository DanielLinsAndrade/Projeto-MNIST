# Parte responsável em lidar com os dados (preparar, carregar) do MNIST.
# Normalização dos dados, gerenciamento dos arquivos compactados,
# gerencia dos dados das imagens e dos rótulos.
#
# Explicação dos imports:
# gzip = para lidar com os arquivos compactados em .gz
# os = para lidar com caminhos dos arquivos
# struct = interpretação dos arquivos MNIST
# numpy = para lidar com manipulação de arrays e outras operações numéricas
import gzip
import os
import struct
import numpy as np


# Função para carregar e normalizar os dados:
# Usando o gzip, irá abrir o arquivo como leitura binária.
# Usando o struct, irá ler os 16 primeiros bytes do cabeçalho do arquivo (f.read(16))
# >IIII significa ler 4 inteiros de 32 bits (4 bytes cada) no formato Big Endian.
# O Big Endian é uma maneira de ler do byte mais importante pro menos importante.
# Os 16 primeiros bytes são identificadores nos arquivos MNIST, sendo:
# 0 ao 3 = identificador do arquivo (_ para ignorar)
# 4 ao 7 = número de imagens (num_imagens)
# 8 ao 11 = número de linhas (linhas)
# 12 ao 15 = número de colunas (colunas)
# sendo eles: identificador do arquivo, número de imagens, linhas e colunas.
# Realiza a leitura dos pixels das imagens e converte isso pra um array numpy.
# Após isso, usando o reshape, se tem um array com os pixels e agora separa nos valores
# para imagens individuais (pixels da imagem, valor de linhas e colunas de forma linear).
# Nomraliza os dados convertendo para valores em float de 0 a 255.
def carregar_imagens(caminho_arquivo):
    with gzip.open(caminho_arquivo, 'rb') as f:
        _, num_imagens, linhas, colunas = struct.unpack(">IIII", f.read(16))
        imagens = np.frombuffer(f.read(), dtype=np.uint8)
        imagens = imagens.reshape(num_imagens, linhas * colunas)
        imagens = imagens.astype(np.float32) / 255.0
        return imagens

# Agora realizando a leitura dos rótulos:
# Novamente fazendo uma leitura binária no arquivo
# Realiza a leitura do arquivo e ignora
# o cabeçalho e o número de rótulos nos 8 primeiros bytes.
# Após isso, cria um array numpy com os índices de cada imagem (rótulo referente a imagem)
# retorna esse array com esses rótulos referentes as imagens.
def carregar_rotulos(caminho_arquivo):
    with gzip.open(caminho_arquivo, 'rb') as f:
        _, _ = struct.unpack(">II", f.read(8))
        rotulos = np.frombuffer(f.read(), dtype=np.uint8)
        return rotulos

# Carregando os arquivos de imagem:
# primeiro padroniza pra vários sistemas o caminho da pasta
# passa pra dentro de cada variável os arrays das funções de rótulos e imagens
# montadas acima.
# retornas essas variáveis pra iniciar o treino.
def carregar_dados_mnist(pasta="dados"):
    X_train = carregar_imagens(os.path.join(pasta, 'train-images-idx3-ubyte.gz'))
    y_train = carregar_rotulos(os.path.join(pasta, 'train-labels-idx1-ubyte.gz'))
    X_test = carregar_imagens(os.path.join(pasta, 't10k-images-idx3-ubyte.gz'))
    y_test = carregar_rotulos(os.path.join(pasta, 't10k-labels-idx1-ubyte.gz'))
    return X_train, y_train, X_test, y_test
