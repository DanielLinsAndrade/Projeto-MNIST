import gzip
import numpy as np
import os
import struct

def carregar_imagens(caminho_arquivo):
    with gzip.open(caminho_arquivo, 'rb') as f:
        magic, num_imagens, linhas, colunas = struct.unpack(">IIII", f.read(16))
        imagens = np.frombuffer(f.read(), dtype=np.uint8)
        imagens = imagens.reshape(num_imagens, linhas * colunas)
        imagens = imagens.astype(np.float32) / 255.0
        return imagens

def carregar_rotulos(caminho_arquivo):
    with gzip.open(caminho_arquivo, 'rb') as f:
        magic, num_rotulos = struct.unpack(">II", f.read(8))
        rotulos = np.frombuffer(f.read(), dtype=np.uint8)
        return rotulos

def carregar_dados_mnist(pasta="dados"):
    X_train = carregar_imagens(os.path.join(pasta, 'train-images-idx3-ubyte.gz'))
    y_train = carregar_rotulos(os.path.join(pasta, 'train-labels-idx1-ubyte.gz'))
    X_test = carregar_imagens(os.path.join(pasta, 't10k-images-idx3-ubyte.gz'))
    y_test = carregar_rotulos(os.path.join(pasta, 't10k-labels-idx1-ubyte.gz'))
    return X_train, y_train, X_test, y_test
