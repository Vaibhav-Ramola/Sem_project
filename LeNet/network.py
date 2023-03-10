import sys

sys.path.append('..')


import numpy as np


from dense import Dense
from conv_layer import Conv
from losses import cross_entropy, cross_entropy_prime
from reshape import Reshape
from activations import TanH, Sigmoid, ReLU

'''
Network:
'''

network = [
    Conv((1, 28, 28), 5, 3),
    ReLU(),
    Conv((5, 26, 26), 10, 3),
    ReLU(),
    Reshape((10, 24, 24), (10 * 24 * 24)),
    Dense(10 * 24 * 24, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid(),
]

def forward(input):
    # input = np.reshape(input, (3, -1, -1))
    for i, layer in enumerate(network):
        input = layer.forward(input)
        # print(f"Layer : {i+1} \t shape : {input.shape}")    # uncomment to know shape after each layer
        # print(input)    # uncomment to see output from each layer
    return input
    
def backward(output_grads, learning_rate):
    for layer in reversed(network):
        output_grads = layer.backward(output_grads, learning_rate)

def loss(y_true, y_pred):
    return cross_entropy(y_pred=y_pred, y_true=y_true)

def loss_prime(y_true, y_pred):
    return cross_entropy_prime(y_true, y_pred)








    