from layer import Layer
import numpy as np
from scipy import signal
import math

class Conv(Layer):
    def __init__(self, input_shape, out_channels, kernel_size, padding=0, stride=1, bias = True):
        super().__init__()
        shape = math.floor(
            (input_shape[1] + 2*padding - kernel_size)/stride
            ) + 1
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.output_shape = (out_channels ,shape, shape)
        
        '''
            I : shape of the input
            P : padding
            K : filter size
            S : stride

            output shape = (I + 2*P - K)/S +1 
        '''
        
        # Initializing the kernels using a normal distribution  
        self.kernel = np.random.randn(
            out_channels, 
            self.in_channels,
            kernel_size, 
            kernel_size)    
        if bias :
            self.bias = np.random.randn(*self.output_shape)


    def forward(self, input):
        self.input = input
        self.input_shape = input.shape
        self.output = np.copy(self.bias)
        '''
        forward propagation is the cross-correlation between 
        each input channel of the image
        and each kernel 
        '''
        # outer loop takes each kernel that on correlation will give an output channel 
        for i, kernel in enumerate(self.kernel):
            # inner loop takes the input and kernel one channel at a time to perform the 
            # correlation operation
            for j in range(self.in_channels):
                self.output[i] += signal.correlate2d(self.input[j], kernel[j], mode='valid')
                # look at : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html
                # for more information on signal.correlation2d(...) function

        
        return self.output

    
    def backward(self, output_grads, learning_rate):
        kernel_grads = np.zeros(self.kernel.shape)
        input_grads = np.zeros(self.input_shape)
        
        '''
        Given:
            Gradient of loss w.r.t. output as output_grads
        To do the backward propagation we need 3 things:
            1.  Gradient of loss w.r.t. kernel(or kernel weights) to update the kernels
            2.  Gradient of loss w.r.t bias(or bias wieghts)  to update bias
            3.  Gradient of loss w.r.t. input so that it can be backward propagated as output_grads for previous layer
        '''
        for out_channel in range(self.out_channels):
            for in_channel in range(self.in_channels):
                kernel_grads[out_channel][in_channel] = signal.correlate2d(self.input[in_channel], output_grads[out_channel], mode='valid')
                # gradient of loss w.r.t. kernels is the valid correlation of input with output_grads
                input_grads[in_channel] += signal.convolve2d(output_grads[out_channel], self.kernel[out_channel][in_channel], mode='full')
                # gradient of loss w.r.t. input is the full convolution of output_grads and kernels
                # To know more about signal.convolve2d, look at
                # Link : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

        # gradient of loss w.r.t. bias is the same as the output_grads

        # Weight update
        self.kernel -= learning_rate * kernel_grads
        self.bias -= learning_rate * output_grads

        # backpropagation
        return input_grads 
        
        
        
