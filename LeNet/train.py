import numpy as np
from network import forward, backward, loss, loss_prime


# Hyperparameters

epochs = 100
lr = 0.01

# Dataset MNIST
# Uncomment below code to test the network on 1000 samples from mnist

# from keras.datasets import mnist

# (x_train, y), _ = mnist.load_data()
# x_train = np.array(x_train[:1000]).reshape(-1, 1, 28, 28)   # testing for only 1000 samples
# y = np.array(y[:1000])  # labels for 1000 samples

# y_train = np.zeros((y.shape[0], 10))

# for i,j in enumerate(y):
#   y_train[i][j] = 1


# for testing random tensors

x_train = np.random.randn(100, 3, 28, 28)
y = np.random.randint(0, 10, size=(100))
y_train = np.zeros((y.shape[0], 10))

for i in range(100):
    y_train[i][y[i]] = 1


def train(lr=0.01, epochs=1000):
    for epoch in range(epochs):
        error = 0;
        for x, y in zip(x_train, y_train):

            #forward prop
            out = forward(x)

            #loss
            error += loss(y, out)

            #backward porp
            backward(loss_prime(y, out), lr)
        error = error/len(x_train)
        print(f"{epoch+1}/{epochs} loss : {error:4f}")
    


if __name__ == '__main__':
    train()
            

