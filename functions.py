"""
関数序盤は非オブジェクト
"""
import numpy as np

# シグモイド


def sigmoid(x):
    return 1 / (1 + np.exp(-x-1e-7))

# シグモイド勾配


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))
