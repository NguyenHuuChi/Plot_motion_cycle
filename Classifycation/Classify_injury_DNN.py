import pandas as pd
import matplotlib.pyplot as plt
import csv
import linecache
import numpy as np
import seaborn as sns
import os
import random

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Model import CNNClassifier,ResNet50,ResNet_important
from special_function import read_text_file ,read_injury_person,create_more_data ,extract_features,extract_different_distance
import tensorflow as tf
tf.config.run_functions_eagerly(True)
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Path_0="D:\\Code\\Python\\summer_research_2\\Classifycation\\4543435"
  


def process_data_2(folder_path):
    files_in_path= os.listdir(folder_path)
    List_processed_file=[]
    for folder in files_in_path :
        if len(folder)>13 and folder[-13:] == "processed.txt" :
            List_processed_file.append(folder)
    
    
    Addition_infor=read_injury_person(folder_path)
    inju=Addition_infor[0]
    Age=Addition_infor[2]
    BIM2=Addition_infor[3]
    Gender=Addition_infor[4]
    # List_columns_to_train=[ "RhipAngZ","RkneeAngZ","RhipMomY","RhipMomZ","RanklePow","RkneePow","LhipAngZ","LkneeAngZ","LhipMomY","LhipMomZ","LanklePow","LkneePow"]
    List_columns_to_train=[
    "RhipAngX", "RhipAngY", "RhipAngZ", "RkneeAngX", "RkneeAngY",
    "RkneeAngZ", "RankleAngX", "RankleAngY", "RankleAngZ",
    "RhipMomX", "RhipMomY", "RhipMomZ", "RkneeMomX", "RkneeMomY",
    "RkneeMomZ", "RankleMomX", "RankleMomY", "RankleMomZ",
    "RgrfX", "RgrfY", "RgrfZ", "RhipPow", "RkneePow", "RanklePow",
    "LhipAngX", "LhipAngY", "LhipAngZ", "LkneeAngX", "LkneeAngY", "LkneeAngZ",
    "LankleAngX", "LankleAngY", "LankleAngZ", "LhipMomX", "LhipMomY", "LhipMomZ",
    "LkneeMomX", "LkneeMomY", "LkneeMomZ", "LankleMomX", "LankleMomY", "LankleMomZ",
    "LgrfX", "LgrfY", "LgrfZ", "LhipPow", "LkneePow", "LanklePow"
        ]
    X=[]
    Y=[]
    # List_columns_to_train=List_columns_to_train1[i*3:i*3+3]
    for i,processed_file in enumerate(List_processed_file) :
        
            
        path_processed =os.path.join(folder_path, processed_file)
        Dataframe= read_text_file(path_processed)
        
        X_shample=extract_different_distance(Dataframe)
        addinfor=[Age[i],BIM2[i],Gender[i]]
        addinfor.extend([-1.]*(101-len(addinfor))) # add all number 0 to other to have an 101 length array
        for j in range(len(X_shample)):
            X_shample[j].append(addinfor)
    
        X.extend([np.array(X_shample[j]) for j in range(len(X_shample))])
        if (inju[i]=="Yes" or inju[i]=="yes"):
            Y.extend([[0.,1.]]*len(X_shample))
        else :
            Y.extend([[1.,0.]]*len(X_shample))
    return X,Y

    
X,Y=process_data_2(Path_0)
seed=42
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,random_state= seed)
scaler=StandardScaler()
transform=scaler.fit(X_train[0])
for i in range(len(X_train)) :
    X_train[i]=transform.transform(X_train[i])
for j in range(len(X_test)) :
    X_test[j]=transform.transform(X_test[j])


X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test=np.array(X_test)
Y_test =np.array(Y_test)

#change the lable into onehot encoding to train


time_step=X_train.shape[1]
input_dim=X_train.shape[2]

def reshape_forconvolution_network(traindata):
    return np.expand_dims(traindata, axis=-1)
X_train=reshape_forconvolution_network(X_train)
X_test=reshape_forconvolution_network(X_test)

input_shape = (13,101, 1)  # Height, width, and single channel
num_classes = 2  # Number of classes for binary classification

# Create an instance of the CNNClassifier
# model1 = CNNClassifier(input_shape, num_classes)
model1=ResNet50(num_classes)
# model1= ResNet_important(num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs =50
batch_size=52
model1.fit(X_train, Y_train, epochs=epochs, batch_size=52, validation_split=0.2)
# Training loop
# for epoch in range(epochs):
#     print(f"Epoch {epoch + 1}/{epochs}")
#     for i in range(0, len(X_train), batch_size):
#         x_batch = X_train[i:i + batch_size]
#         y_batch = Y_train[i:i + batch_size]

#         with tf.GradientTape() as tape:
#             # Forward pass
#             predictions = model1(x_batch)
#             loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, predictions))

#         # Compute gradients with unconnected_gradients=tf.UnconnectedGradients.ZERO
#         gradients = tape.gradient(loss, model1.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

#         # Apply gradients
#         optimizer.apply_gradients(zip(gradients, model1.trainable_variables))

#     # Validation or other logging here
predictions = model1.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = np.argmax(Y_test, axis=1)  # Assuming Y_test is one-hot encoded labels

accuracy = np.mean(predicted_labels == true_labels)
print("Accuracy:", accuracy)



# Model= Sequential()

#----------------Simple LSTM for sequence Classification-----------------
# print("-----------------Simple LSTM for sequence Classification--------------")
# Model.add(LSTM(100,input_shape=(time_step,input_dim)))
# Model.add(Dense(2,activation= "softmax"))


#---------------LSTM for Sequence Classification with Dropout----------------
# Model.add(LSTM(100,input_shape=(time_step,input_dim)))
# Model.add(Dropout(0.2))
# Model.add(Dense(2,activation= "softmax"))

#------------Bidirectional LSTM for Sequence Classification----------------
# Model.add(Bidirectional(LSTM(50,input_shape=(time_step,input_dim))))
# Model.add(Dense(10,activation= "sigmoid"))
# Model.add(Dropout(0.3))
# Model.add(Dense(2,activation= "softmax"))


#----------LSTM and Convolutional Neural Network for Sequence Classification------------------

# The input is (batch_size, sequence_length, input_channels)--> out put is (batch_size, sequence_length, filters) filter is what we want in the output -> in the layer
# Model.add( Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# Model.add(LSTM(100,input_shape=(time_step,input_dim)))
# Model.add(Dropout(0.2))
# Model.add(Dense(10,activation= "softmax"))
# Model.add(Dropout(0.3))
# Model.add(Dense(2,activation="softmax"))


# Batch_size=32

# step_per_epoch= len(X_train)// Batch_size
# step_per_epoch=max(1, step_per_epoch)

# Model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
# early_stopping=  EarlyStopping(monitor='val_accuracy',patience=10, restore_best_weights=True) # if it decrease 3 time continuely , it will stop trainning and  recover the weight
# # Model.fit(X_train,Y_train, epochs= 100, batch_size=Batch_size, validation_data=(X_val,Y_val),callbacks=[early_stopping],steps_per_epoch=step_per_epoch) # we need  validation dataset if we want to use EarlyStopping 
# Model.fit(X_train,Y_train, epochs= 100, batch_size=1,validation_split=0.2,callbacks=[early_stopping]) # Train all data for limited dataset
# # Model.fit(X_train,Y_train, epochs= 100, batch_size=1)
# scores=Model.evaluate(X_test,Y_test,verbose =0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
