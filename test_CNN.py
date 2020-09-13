from keras.models import load_model
import matplotlib.image as img
import numpy as np


gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])

model = load_model('weights2.model')
model.summary()

for i in range(26):
    image = gray(img.imread("new_test_data/" + chr(ord('A') + i) + ".png"))

    image = image.reshape(1, 28, 28, 1)

    pixels = image.reshape((28, 28))

    result = model.predict(image)
    prediction = np.argmax(result, axis=1)

    print("prediction: ", prediction, chr(ord('A') + prediction[0]), "%.2f%%" % (result[0][prediction[0]] * 100), "+" if prediction[0] == i else "-")

