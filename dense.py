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

    '''
    Forward propagation function
    '''
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.bias)

        for out_feature in range(self.out_features):
            for in_feature in range(self.in_features):
                self.output[out_feature] += self.weights[out_feature][in_feature] * input[in_feature]

        return self.output
    
    '''
    Back-propagation function
    '''
    def backward(self, output_grad, learning_rate):

        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)
        self.input_grads = np.zeros(self.in_features)

        for out_feature in range(self.out_features):
            for in_feature in range(self.in_features):
                self.weights[out_feature][in_feature] = output_grad[out_feature] * self.input[in_feature]
                self.input_grads[in_feature] += self.weights[out_feature][in_feature] * output_grad[out_feature]

        self.bias_grad = output_grad

        # Weight update
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

        return  self.input_grads
