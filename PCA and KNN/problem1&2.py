import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
from collections import Counter 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

file= pd.read_csv("Iris.csv")
x= file.drop("Species",axis=1)
y= file["Species"]
    
description=x.describe()  

for i in (x.columns):
    outliers=[]
    lower_quartile=description[i]['25%']
    upper_quartile=description[i]['75%']
    inter_quartile_range = upper_quartile-lower_quartile
    for j in range(len(x[i])):
        median=description[i]['50%']
        if(not (lower_quartile - 1.5*inter_quartile_range < x.at[j,i] < upper_quartile +1.5*inter_quartile_range)):
            x.at[j,i]=median
    
X=x
X_bar= X-X.mean()
print(X_bar)
C=X_bar.transpose().dot(X_bar)
e_value,e_vector= np.linalg.eig(C)
print("EigenValues: ",e_value,"\n")
print("EigenVectors:\n",e_vector,"\n")
Q= e_vector[:,:2]
print(Q)

X_cap=X.dot(Q)
X_reduce=X_cap
X_reduce.columns=['x0','x1']

soa = np.array([[X_reduce['x0'].mean(), X_reduce['x1'].mean(), e_value[0] * e_vector[0][0], e_value[0] * e_vector[1][0]]])
soa1 = np.array([[X_reduce['x0'].mean(), X_reduce['x1'].mean(), e_value[1] * e_vector[0][1], e_value[1] * e_vector[1][1]]])
x0, y0, u0, v0 = zip(*soa)
x1, y1, u1, v1 = zip(*soa1)

print(X_reduce)
print("Reduced Data:\n",X_reduce)
plt.scatter(X_reduce['x0'],X_reduce['x1'],color='red',edgecolors='black',label='Original Data Points')
plt.title("Plot of Reduced data and Eigen directions")
plt.xlabel("PCA 1---->",fontsize=13)
plt.ylabel("PCA 2---->",fontsize=13)
plt.quiver(x0, y0, u0, v0, angles='xy', scale_units='xy',color='blue', scale=100, width=0.005,)
plt.quiver(x1, y1, u1, v1, angles='xy', scale_units='xy',color='blue', scale=25, width=0.005, label='Eigen Vectors')
plt.grid(color='limegreen',linestyle='--',linewidth=0.7)
plt.legend()
plt.show()

X_dot=np.dot(X_cap,Q.T)
X_reconstructed=pd.DataFrame(X_dot,columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
print("Reconstructed Data:\n",X_reconstructed)

rmse=[]
for l in (X.columns):
    num=0
    for n in range(len(X_reconstructed)):
        num+=(X.at[n,l]-X_reconstructed.at[n,l])**2
    rmse.append((num/(len(X_reconstructed)-1))**0.5)

print("RMSE: ",rmse)
print("\n")

X_reduce_train,X_reduce_test,y_train,y_test= sklearn.model_selection.train_test_split(X_reduce,y,random_state=104,test_size=0.20,shuffle=True)
K=5
actual_predicted=[]
for k in range(len(X_reduce_test)):
    distance=[]
    for z in range(len(X_reduce_train)):
        distance_=np.sqrt(np.sum((X_reduce_test.iloc[k]-X_reduce_train.iloc[z])**2))
        distance.append([distance_,y_train.iloc[z]])
    
    distance=sorted(distance,key=lambda x:x[0])
    top_K_neighbors=distance[:K]
    predicted_class = Counter(top_K_neighbor[1] for top_K_neighbor in top_K_neighbors).most_common(1)[0][0]
    actual_predicted.append((y_test.iloc[k],predicted_class))
    
matrix=np.array(actual_predicted)
print("Matrix of Actual and Predicted class labels: \n")
print(matrix,"\n")

actual=matrix[:,0]
predicted=matrix[:,1]
c_matrix=confusion_matrix(actual,predicted)
print("Confusion Matrix: ")
print(c_matrix)

ConfusionMatrixDisplay(c_matrix,display_labels=['Class 1','Class 2','Class 3']).plot()
plt.show()
