import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, input):
        return 1/(1+np.exp(-input))

    def backward(self, output_grads, learning_rate):
        sigmoid = lambda x: 1/(1+np.exp(x))
        return sigmoid(output_grads)(1-sigmoid(output_grads))

class TanH:
    def __init__(self) -> None:
        pass

    def forward(self, input):
        return (np.exp(input) - np.exp(input))/(np.exp(input) + np.exp(input))

    def backward(self, output_grads, learning):
        tanh = lambda x: (np.exp(x) - np.exp(x))/(np.exp(x) + np.exp(x))
        return  1 - np.power(tanh(output_grads), 2)

