import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x));

    def forward(self, input):
        return 1/(1+np.exp(-input))

    def backward(self, output_grads, learning_rate):
        x = self.sigmoid(output_grads)
        x = x*(1-x)
        return x

class TanH:
    def __init__(self) -> None:
        pass

    def forward(self, input):
        return (np.exp(input) - np.exp(input))/(np.exp(input) + np.exp(input))

    def backward(self, output_grads, learning):
        tanh = lambda x: (np.exp(x) - np.exp(x))/(np.exp(x) + np.exp(x))
        return  1 - np.power(tanh(output_grads), 2)

'''
This ReLU is specific for use in CNNs
'''
class ReLU:
    def __init__(self) -> None:
        pass

    def forward(self, input):
        self.input = input
        self.output = np.zeros(input.shape)
        in_shape = input.shape
        for i in range(in_shape[0]):
            for j in range(in_shape[1]):
                for k in range(in_shape[2]):
                    if self.input[i][j][k]>0:
                        self.output[i][j][k] = self.input[i][j][k]

        return self.output

    def backward(self, output_grads):
        out_shape = output_grads.shape
        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                for k in range(out_shape[2]):
                    if output_grads[i][j][k]>0: 
                        output_grads[i][j][k] = 1 

        return output_grads
