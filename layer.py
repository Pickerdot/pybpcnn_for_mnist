import numpy as np
import functions as fn


class Loss:
    def __init__(self):
        self.Loss = None
        self.dout = None

    def forward(self, out, t):
        self.Loss = 1/2 * np.sum((out - t)**2)
        self.dout = out - t
        return self.Loss

    def backward(self):
        return self.dout


class hidden_layer:
    def __init__(self, W, b, lr):
        # 入力側からの重みと自ニューロンへのバイアスの格納
        self.W = W
        self.b = b
        # W、bに対する勾配の入れ物
        self.W_grad = None
        self.b_grad = None
        # 活性値と出力を格納
        self.y = None
        self.z = None
        # 入力の格納
        self.x = None
        # 学習率の格納
        self.lr = lr

    def forward(self, x):
        # x格納
        self.x = x
        # y：活性値
        self.y = np.dot(x, self.W) + self.b
        # Z：ニューロンの出力
        self.z = fn.sigmoid(self.y)
        return self.z

    def backward(self, dz):
        # 出力部の逆伝搬（シグモイド版）
        dy = fn.sigmoid_grad(self.z) * dz
        #print("dy:", dy)
        #print("x.T:", self.x.T)
        self.b_grad = dy
        self.W_grad = np.dot(self.x.T, dy)
        dx = np.dot(dy, self.W.T)
        #print("dw:", self.W_grad)
        #print("W:", self.W)
        W_tmp = self.W
        #print("dW", np.sum(self.W_grad))
        # オプティマイザーによりself.W、self.bの値を更新
        self.W = self.W - self.lr * self.W_grad
        self.b = self.b - self.lr * self.b_grad

        #print("Wn-W", np.sum(self.W - Wn))

        #print("W:", self.W)

        return dx
