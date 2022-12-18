import numpy as np

eps = 10e-10

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, input):
        input = self.normalize(input)   # Normalization
        return 1/(1+np.exp(-input) + eps)

    def backward(self, output_grads, learning_rate):
        sigmoid = lambda x: 1/(1+np.exp(x))
        return sigmoid(output_grads)(1-sigmoid(output_grads))

class TanH:
    def __init__(self) -> None:
        pass
    
    def normalize(self, input):
        return input/np.max(np.abs(input))

    def forward(self, input):
        input = self.normalize(input)   # Normalizing the input
        return (np.exp(input) - np.exp(-input))/(np.exp(input) + np.exp(-input) + eps)        # fixed the tanh activation function
        # added eps to denominator to prevent devision by 0
    
    def backward(self, output_grads, learning):
        output_grads = self.normalize(output_grads)     # Normalizing the output gradients
        tanh = lambda x: (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x) + eps)              # same fix of -ve sign here
        return  1 - np.power(tanh(output_grads), 2)

'''
This ReLU is specific for use in CNNs
'''
class ReLU:
    def __init__(self) -> None:
        pass


    def normalize(self, input):
        return input/np.max(np.abs(input))

    def forward(self, input):
        input = self.normalize(input)           # Normalizing incomming inputs
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
        output_grads = self.normalize(output_grads)         # Normalizing incomming gradients
        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                for k in range(out_shape[2]):
                    if output_grads[i][j][k]>0: 
                        output_grads[i][j][k] = 1 

        return output_grads
