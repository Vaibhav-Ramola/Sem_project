import numpy as np


# Hyperparameters

epochs = 100
lr = 0.01

x_train = np.random.randn(100, 3, 28, 28)
y = np.random.randint(0, 10, size=(100))
y_train = np.zeros((y.shape[0], 10))

for i in range(100):
    y_train[i][y[i]] = 1

print(y_train)