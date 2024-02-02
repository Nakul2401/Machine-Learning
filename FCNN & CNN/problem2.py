import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from keras.layers import Dense , Conv2D, Flatten, MaxPooling2D


def data_loader(path_train, path_test):
  train_list=os.listdir(path_train)

  # Number of classes in the dataset
  num_classes=len(train_list)

  # Empty lists for loading training and testing data images as well as corresponding labels
  x_train=[]
  y_train=[]
  x_test=[]
  y_test=[]

  # Loading training data
  for label, folder in enumerate(train_list):
    path1 = path_train+'/'+str(folder)
    images = os.listdir(path1)
    for file in images:
      path2 = path1+'/'+str(file)

      # Read the image form the directory
      img = cv2.imread(path2)

      # Append image to the train data list
      x_train.append(img)

      # Append class-label corresponding to the image
      y_train.append(label)
  
    # Loading testing data
    path1 = path_test+'/'+str(folder)
    images = os.listdir(path1)
    
    for file in images:
      path2=path1+'/'+str(file)
      img = cv2.imread(path2)
      x_test.append(img)
      y_test.append(label)
  
  # Convert lists into numpy arrays
  x_train=np.asarray(x_train)
  y_train=np.asarray(y_train)
  x_test=np.asarray(x_test)
  y_test=np.asarray(y_test)
  return x_train,y_train,x_test,y_test

path_train = 'cifar-3class-data/train'
path_test = 'cifar-3class-data/test'
X_train,y_train,X_test,y_test = data_loader(path_train, path_test)
print("Loading Done")

X_train,X_validation, y_train,y_validation=train_test_split(X_train,y_train,test_size=0.10)

X_train=X_train/255
X_validation=X_validation/255

model=Sequential([Conv2D(64,(3,3),activation='relu',padding='same'),
                  Conv2D(64,(3,3),activation='relu',padding='same'),
                  MaxPooling2D(pool_size=(2,2),padding='valid'),
                  Conv2D(128,(3,3),activation='relu',padding='same'),
                  Conv2D(128,(3,3),activation='relu',padding='same'),
                  MaxPooling2D(pool_size=(2,2),padding='valid'),
                  Flatten(),
                  Dense(512,activation='relu'),
                  Dense(100,activation='relu'),
                  Dense(3,activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

info=model.fit(X_train,y_train,validation_data=(X_validation,y_validation),epochs=50,batch_size=200)

model.save_weights('CNN.h5')

plt.figure(figsize=(8,6))
plt.plot(info.history['accuracy'],label='Training Accuracy',color='black')
plt.plot(info.history['val_accuracy'],label='Validation Accuracy',color='red')
plt.xlabel("Epochs--->")
plt.ylabel("Accuracies--->")
plt.title("Epoch-wise training and validation accuracies, CNN")
plt.legend()
plt.show()

X_test=X_test/255

model.load_weights('CNN.h5')

probabilities=model.predict(X_test)
print("Predicted Probablities")
print(probabilities,'\n')

predicted_class=np.argmax(probabilities,axis=1)
print("Predicted Class:")
print(predicted_class,'\n')

c_matrix=confusion_matrix(y_test,predicted_class)
print("Confusion Matrix:")
print(c_matrix)

ConfusionMatrixDisplay(c_matrix).plot()
plt.show()

evaluation=model.evaluate(X_test,y_test)
print("Test Loss:",evaluation[0])
print("Test Accuracy:",evaluation[1]*100,'%')


