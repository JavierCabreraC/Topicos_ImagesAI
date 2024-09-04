import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# CONSTANTES
DATA_DIR = 'imgsBilletes'
IMG_HEIGHT = 224
IMG_WIDTH = 224 
BATCH_SIZE = 32
EPOCHS = 20

# Función para el modelo convolucional
def build_currency_model(num_classes):
    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                             include_top=False,
                             weights='imagenet')
    
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Función para preparar la dataset
def prepare_currency_dataset():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training')
    
    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation')
    
    return train_generator, validation_generator

# Entrenamiento del modelo
def train_currency_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )
    return history

# Función para predecir la clasificación de una imagen
def predict_currency(model, image_path, class_indices):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    class_names = list(class_indices.keys())
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    print(f"La imagen es de un billete de  <<{predicted_class}>> , con una confianza del {confidence:.2f} %.")

# Ejecución
if __name__ == "__main__":
    # Preparar la dataset
    train_generator, validation_generator = prepare_currency_dataset()
    
    # Obtener la cantidad de clases
    num_classes = len(train_generator.class_indices)
    
    # Construir y entrenar el modelo
    model = build_currency_model(num_classes)
    history = train_currency_model(model, train_generator, validation_generator)
    
    # Guardar el modelo
    model.save('currency_recognition_model.h5')
    
    # Guardar los índices de las clases
    with open('currency_class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    
    # Reemplzar o modificar el path para cargar la imagen para probar
    path = 'test/200/IMG_20240901_213558_196.jpg'
    predict_currency(model, path, train_generator.class_indices)

