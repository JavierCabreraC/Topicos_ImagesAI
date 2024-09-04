# Script para probar un modelo ya compilado

import tensorflow as tf
import numpy as np
import json

class CurrencyRecognizer:
    def __init__(self, model_path, class_indices_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        self.class_names = {v: k for k, v in self.class_indices.items()}

    def predict_currency(self, image_path):
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(224, 224)  # Estas dimensiones deben coincider con el modelo
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalización

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = self.class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        return predicted_class, confidence

# Uso
if __name__ == "__main__":
    recognizer = CurrencyRecognizer('currency_recognition_model.h5', 'us_currency_class_indices.json')
    
    # Testeo:
    image_path = 'prueba/10v1.jpg'
    print('La imagen de prueba es: ' + image_path)
    predicted_denomination, confidence = recognizer.predict_currency(image_path)
    
    print(f"La imagen es de un billete de  <<{predicted_denomination}>> , con una confianza del {confidence:.2f} %.")

