import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense , Conv2D, Flatten, MaxPooling2D


def data_loader(path_train, path_test):
  train_list=os.listdir(path_train)
  # print(train_list)
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
print("Size of TRAIN data:",X_train.shape)

fig,axs = plt.subplots(1,4,figsize=(15, 5))
for i in range(4):
  index=np.random.randint(len(X_train))
  axs[i].imshow(X_train[index])
  axs[i].set_title(f'Class: {y_train[index]}')
  axs[i].axis('off')

plt.show()

X_train,X_validation, y_train,y_validation=train_test_split(X_train,y_train,test_size=0.10)

print("Size of new TRAIN data:",X_train.shape)
print("Size of Validation set:",X_validation.shape)

vector_length=X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
X_train=X_train.reshape(X_train.shape[0],vector_length).astype('float32')
X_validation=X_validation.reshape(X_validation.shape[0],vector_length).astype('float32')

X_train=X_train/255
X_validation=X_validation/255

model=Sequential([Dense(256,activation='relu'),
                  Dense(128,activation='relu'),
                  Dense(64,activation='relu'),
                  Dense(3,activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

info=model.fit(X_train,y_train,validation_data=(X_validation,y_validation),epochs=500,batch_size=200)

plt.figure(figsize=(8,6))
plt.plot(info.history['accuracy'],label='Training Accuracy',color='black')
plt.plot(info.history['val_accuracy'],label='Validation Accuracy',color='red')
plt.xlabel("Epochs--->")
plt.ylabel("Accuracies--->")
plt.title("Epoch-wise training and validation accuracies, FCNN")
plt.legend()
plt.show()

X_test=X_test.reshape(X_test.shape[0],vector_length).astype('float32')
X_test=X_test/255

#Testing the model:
evaluation=model.evaluate(X_test,y_test)
print("Test Loss:",evaluation[0])
print("Test Accuracy:",evaluation[1]*100,'%')







