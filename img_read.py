from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import load_model
import matplotlib.image as img
import numpy as np
from matplotlib import pyplot as plt


gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])

model = load_model('weights2.model')
model.summary()
image = gray(img.imread("letter.png"))



image = image.reshape(1, 28, 28, 1)

pixels = image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

print(image.shape)
#print(image)
#input()


result = model.predict(image)
prediction = np.argmax(result, axis=1)

print("reuslt: ", result)
print("prediction: ", prediction, chr(ord('A') + prediction[0]), "%.2f%%" % (result[0][prediction[0]] * 100))


