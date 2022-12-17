import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = np.random.randn(out_features, in_features)
        if bias:
            self.bias = np.random.randn(out_features)
        else:
            self.bias = np.zeros(out_features)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.bias)

        for out_feature in range(self.out_features):
            for in_feature in range(self.in_features):
                self.output[out_feature] += self.weights[out_feature][in_feature] * input[in_feature]

        return self.output

    

    def backward(self, output_grad, learning_rate):
        
        return
