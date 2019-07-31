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


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

        return params

    def change_lr(self, lr):
        self.lr = lr
