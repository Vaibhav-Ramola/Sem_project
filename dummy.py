import numpy as np

a = np.random.randn(3, 3, 3)

b = [a if a>0 else 0]

print(a)
print(b)