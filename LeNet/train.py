import numpy as np
from network import forward, backward, loss, loss_prime


# Hyperparameters

epochs = 100
lr = 0.01

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
        print(f"{epoch}/{epochs} loss : {error}.:4f")
    


if __name__ == '__main__':
    train()
            

