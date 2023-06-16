import cv2
import numpy as np
from keras.models import load_model

# Cargar el modelo entrenado
model = load_model('actionwithcnn.h5')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Dimensiones de la imagen de entrada del modelo
img_width, img_height = 128, 128

# Mapeo de las clases de salida del modelo a acciones del robot
actions = np.array(['ven', 'para', 'sientate'])

# Función para preprocesar la imagen
def preprocess_image(img):
    img = cv2.resize(img, (img_width, img_height))
    img = np.array(img)
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=0)
    return img

# Loop principal
while True:
    # Capturar una imagen de la cámara
    ret, frame = cap.read()
    
    # Preprocesar la imagen
    processed_image = preprocess_image(frame)
    
    # Realizar la predicción
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    
    # Obtener la acción correspondiente a la predicción
    action = actions[predicted_class]
    
    # Mostrar la imagen en la pantalla
    cv2.imshow('Capturando Imagen', frame)
    
    # Esperar a que se presione la tecla 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()