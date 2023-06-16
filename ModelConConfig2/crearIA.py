#Librerias e imports
import tensorflow
import keras
import sklearn
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization,  TimeDistributed, Conv2D, MaxPooling2D, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
#Vuelvo a declarar las varibles de actions sequences y datapath, en lugar de importarlas del otro archivo, para evitar lanzarlo mas de una vez porq eso hara que se sobreescriban los datos ya recogidos 
DATA_PATH = os.path.join('MP_Data_100')
actions = np.array(['ven', 'para', 'sientate'])
sequence_length = 30

#ORDENAR LOS DATOS RECOGIDOS DE FORMA QUE TENSORFLOW PUEDA TRABAJAR CON ELLOS
label_map = {label:num for num, label in enumerate(actions)} # ven:0, para:1, sientate:2
sequences, labels = [], [] #Secuencias actuan como X, labels como Y de una funcion, la IA ajustara el resto de parametros para hacer que cada X lleve a su Y correspondiente
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length): #Para cada uno de los frames
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))) #Cargar los datos de numpy de los directorios
            window.append(res) #Mete cada frame separado en una ventana conjunta
        sequences.append(window) #Junta todos los videos/ventanas en un vector que contiene todos los videos de manea ordenada
        labels.append(label_map[action]) #Junta todos los nombres de la actions en un vector de manera ordena
#RESULTADO --> secuences[0] = video0Ven , labels[0] = 0 , ... , secuences[30] = video0Para , labels[30] = 1 , ... , 

X = np.array(sequences)
Y = to_categorical(labels).astype(int) #De int a binario
#Particiones de crear modelo y testear modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#LSTM 
#logs de como se esta entrenando la IA en tiempo real, se pueden ver en web
#log_dir = os.path.join('Logs_100_Config2')
#tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(actions.shape[0], activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test)) #callbacks=[tb_callback])
model.save('actionConConfig2.h5')
model.summary()

yhat = model.predict(X_test)
ytrue = np.argmax(Y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
confusion = multilabel_confusion_matrix(ytrue, yhat)
accuracy = accuracy_score(ytrue, yhat)
print(confusion)
print(accuracy)