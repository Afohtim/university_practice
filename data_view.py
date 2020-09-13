import numpy as np
from matplotlib import pyplot as plt

print("Loading dataset...")
dataset = np.loadtxt('data/A_Z Handwritten Data.csv', delimiter=',')

print("Preparing data")

X = dataset[:,0:784]
Y = dataset[:,0]


X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')
X = X / 255


while True:
    n = int(input("AGANE"))

    if n == -1:
        n = np.random.randint(0, X.shape[0])
    print(Y[n])
    image = X[n]
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
