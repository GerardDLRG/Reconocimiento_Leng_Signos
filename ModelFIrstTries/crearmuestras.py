#LIBRERIAS Y DEPENDENCIAS
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

#CONSEGUIR PUNTOS DE REFERENCIA Y VISUALIZARLOS CON MEDIAPIPE Y CV2

mp_holistic = mp.solutions.holistic # Para que mediapipe leea los puntos de referencia desde la camara
mp_drawing = mp.solutions.drawing_utils # Para ver por pantalla los puntos de referencia que mediapìpe esta leyendo en la variable anterior

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # conversión de color del frame leido de la camara a uno con el que mediapipe pueda trabajar
    image.flags.writeable = False                  # Freezeo de la imagen para hacer la predicción
    results = model.process(image)                 # Predicción 
    image.flags.writeable = True                   # Descongelar imagen una vez hecha la prediccón 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Desacemos la conversión de color anterior
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Pinta las conexiones de la pose 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Pinta las conexiones sobre la imagen NO por pantalla para pintar por pantala se usa matplotlib
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Pinta las conexiones aobre la imagen NO por pantalla

#EXTRAER LOS PUNTOS DE REFERENCIA Y CONEXIONES DE MEDIAPIPE PARA ANALISIS
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

#CREAR LAS CARPETAS PARA ALMACENAR LOS DATOS
# Direccion para guardar la carpetas
DATA_PATH = os.path.join('MP_Data') 
# Acciones que la IA ha de detectar
actions = np.array(['ven', 'para', 'sientate'])
# 30 videos de 30 frames de duracion para cada accion 
no_sequences = 30
sequence_length = 30
start_folder = 30
#BUCLE PARA CREAR LAS CARPETAS AUTOMATICAMENTE
for action in actions: 
    for sequence in range(0,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

#CAPTURA DE DATOS USANDO LAS FUNCIONES ANTERIORES, CON MEDIAPIPE Y OPENCV
#Declarar que camara usar, ir probrando en caso de que la camara deseada no este en la localización 0
cap = cv2.VideoCapture(0)
# Declarar modelo de MediaPipe 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Bucle a traves de las diferentes acciones definidas antes (se pueden añadir mas acciones al vector sin causar fallos)
    for action in actions:
        # Bucle para capturar numero de videos declarado en variable no_secuences
        for sequence in range(no_sequences):
            # Bucle para capturar los frames en sequence_lenght, para cada video
            for frame_num in range(sequence_length):

                # Leer camara
                ret, frame = cap.read()
                # Detectar landmarks con MediaPipe
                image, results = mediapipe_detection(frame, holistic)
                # Pintar las detecciones por pantalla para que queda mejor visualmente
                draw_landmarks(image, results)
                # Esperars para la captura de datos
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                # Extraer los datos en las carpetas declaradas antes
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    #Al acabar cerrar todas las instancias de openCV y la camara                
    cap.release()
    cv2.destroyAllWindows()
