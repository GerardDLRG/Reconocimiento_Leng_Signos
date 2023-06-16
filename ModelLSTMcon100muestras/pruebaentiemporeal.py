import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from scipy import stats
import tensorflow
import keras
import sklearn
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import time


DATA_PATH = os.path.join('MP_Data')
actions = np.array(['ven', 'para', 'sientate'])
sequence_length = 30
mp_holistic = mp.solutions.holistic # Para que mediapipe leea los puntos de referencia desde la camara
mp_drawing = mp.solutions.drawing_utils # Para ver por pantalla los puntos de referencia que mediapìpe esta leyendo en la variable anterior
colors = [(245,117,16), (117,245,16), (16,117,245)]

def forward(speed=100):
    #dataCMD = json.dumps({'var': "move", 'val': 1})
    #ser.write(dataCMD.encode())
    print('robot-forward')


def steadyMode():
    #dataCMD = json.dumps({'var': "funcMode", 'val': 1})
    #ser.write(dataCMD.encode())
    print('robot-steady')


def handShake():
    #dataCMD = json.dumps({'var': "funcMode", 'val': 3})
    #ser.write(dataCMD.encode())
    print('robot-handshake')

last_action_time = time.time()
min_time_between_actions = 3  # Minimum time difference in seconds

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # conversión de color del frame leido de la camara a uno con el que mediapipe pueda trabajar
    image.flags.writeable = False                  # Freezeo de la imagen para hacer la predicción
    results = model.process(image)                 # Predicción 
    image.flags.writeable = True                   # Descongelar imagen una vez hecha la prediccón 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Desacemos la conversión de color anterior
    return image, results

def draw_styled_landmarks(image, results): 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
#EXTRAER LOS PUNTOS DE REFERENCIA Y CONEXIONES DE MEDIAPIPE PARA ANALISIS
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('actionCon100.h5')
    
# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.7

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not receive frame.")
            break

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            detected_action = actions[np.argmax(res)]
            print(detected_action)
            predictions.append(np.argmax(res))

             
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    current_time = time.time()
                    if len(sentence) > 0 and detected_action != sentence[-1] and (current_time - last_action_time) > min_time_between_actions:
                        if detected_action == 'ven':
                            forward()
                        elif detected_action == 'para':
                            steadyMode()
                        elif detected_action == 'sientate':
                            handShake()
                        last_action_time = current_time
            
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()