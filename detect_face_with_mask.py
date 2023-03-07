import cv2
import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

video = cv2.VideoCapture(0)

while True:

    check,frame = video.read()

    # Modificar los datos de entrada al:

    # 1. Redimensionar la imagen

    img = cv2.resize(frame,(224,224))

    # 2. Convertir la imagen en una matriz Numpy e incrementar la dimensión

    test_image = np.array(, dtype=np.float32)
    test_image = np.expand_dims()

    # 3. Normalizar la imagen
    normalised_image = 

    # Predecir el resultado
    prediction = model.predict()

    print("Predicción: ", prediction)
        
    cv2.imshow("Resultado",frame)
            
    key = cv2.waitKey(1)

    if key == 32:
        print("Cerrando")
        break

video.release()