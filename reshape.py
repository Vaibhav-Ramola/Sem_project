import numpy as np

class Reshape:
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, input):
        return np.reshape(input, self.out_shape)

    def backward(self, output_grads,learning_rate):
        return np.reshape(output_grads, self.in_shape)
