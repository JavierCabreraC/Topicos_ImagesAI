# 1 - Dependencias:

import os
import tensorflow as tf
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

# 2 - Limpiar imágenes:

data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'png', 'bmp']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            os.remove(image_path)

# 3 - Cargar datos:

data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# 4 - Escalar Datos:

data = data.map(lambda x,y: (x/255, y))

data.as_numpy_iterator().next()

# 5 - Chunking Datos:

train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# 6 - Construcción del Modelo:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

num_classes = 5
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

model.summary()

# 7 - Entrenamiento:

logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=21, validation_data=val, callbacks=[tensorboard_callback])

# 8 - Gráficos de Rendimiento

# fig = plt.figure()
# plt.plot(hist.history['loss'], color='teal', label='perdida')
# plt.plot(hist.history['val_loss'], color='orange', label='val_perdida')
# fig.suptitle('Pérdida', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()

# fig = plt.figure()
# plt.plot(hist.history['accuracy'], color='teal', label='exactitud')
# plt.plot(hist.history['val_accuracy'], color='orange', label='val_exactitud')
# fig.suptitle('Exactitud', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()

# 9 - Evaluación

from tensorflow.keras.metrics import CategoricalAccuracy
acc = CategoricalAccuracy()

# print(pre.result(), re.result(), acc.result())
print(acc.result())

# 10 - Testeo de Imágenes:

img = cv2.imread('desayuno.jpg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))

predicted_class = np.argmax(yhat, axis=1)
class_names = ['cien', 'cincuenta', 'diez', 'doscientos', 'veinte']  # Replace with your actual class names
print(class_names)
print(f'La clase es: {class_names[predicted_class[0]]}')

