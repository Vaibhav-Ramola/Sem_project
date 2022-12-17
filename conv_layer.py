from layer import Layer
import numpy as np
from scipy import signal
import math

class Conv(Layer):
    def __init__(self, input_shape, out_channels, padding, kernel_size, stride, bias = True):
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        shape = math.floor(
            (input.shape[0] + 2*self.padding - self.kernel_size)/self.stride
            ) + 1
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

    
    def backward(self, output_grad, learning_rate):
        

        return super().backward(output_grad, learning_rate)
        
        
        return super().forward(input)
