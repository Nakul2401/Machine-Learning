import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

file= pd.read_csv("iris_test.csv")
file_=pd.read_csv("iris_train.csv")
file=file.drop("Unnamed: 0",axis=1)
file_=file_.drop("Unnamed: 0",axis=1)
x= file.drop("Species",axis=1)
y= file["Species"]
x1= file_.drop("Species",axis=1)
y1= file_["Species"]

X=x
X_bar= X-X.mean()
C=X_bar.transpose().dot(X_bar)
e_value,e_vector= np.linalg.eig(C)
Q= e_vector[:,:1]
X_cap=X_bar.dot(Q)
X_reduce_test=X_cap
X_reduce_test.columns=['x0']

X=x1
X_bar= X-X.mean()
C=X_bar.transpose().dot(X_bar)
e_value,e_vector= np.linalg.eig(C)
Q= e_vector[:,:1]
X_cap=X_bar.dot(Q)
X_reduce_train=X_cap
X_reduce_train.columns=['x0']

frames=[X_reduce_test,y]
frames2=[X_reduce_train,y1]
X_reduce_tested=pd.concat(frames,axis=1, join='inner')
X_reduce_trained=pd.concat(frames2,axis=1, join='inner')

def meann(x):
    return sum(x)/len(x)
def stD(d):
    mean = meann(d)
    squared_diff = [(x - mean) ** 2 for x in d]
    return (sum(squared_diff) / len(d))** 0.5

Class_1=X_reduce_trained[X_reduce_trained["Species"]=='Iris-setosa']
Class_2=X_reduce_trained[X_reduce_trained["Species"]=='Iris-virginica']
Class_3=X_reduce_trained[X_reduce_trained["Species"]=='Iris-versicolor']

list1=[]
for i in range(len(X_reduce_tested)):
    Gaussian_p_1=(1/((stD(Class_1["x0"]))*(2*np.pi)**0.5)*np.exp((-0.5*((X_reduce_tested['x0'][i]-meann(Class_1["x0"]))/(stD(Class_1["x0"])))**2)))
    Gaussian_p_2=(1/((stD(Class_2["x0"]))*(2*np.pi)**0.5)*np.exp((-0.5*((X_reduce_tested['x0'][i]-meann(Class_2["x0"]))/(stD(Class_2["x0"])))**2)))
    Gaussian_p_3=(1/((stD(Class_3["x0"]))*(2*np.pi)**0.5)*np.exp((-0.5*((X_reduce_tested['x0'][i]-meann(Class_3["x0"]))/(stD(Class_3["x0"])))**2)))
    posterior_1=(Gaussian_p_1)*(len(Class_1)/len(X_reduce_trained))
    posterior_2=(Gaussian_p_2)*(len(Class_2)/len(X_reduce_trained))
    posterior_3=(Gaussian_p_3)*(len(Class_3)/len(X_reduce_trained))
    if posterior_1>posterior_2 and posterior_1>posterior_3:
        list1.append("Iris-setosa")
    elif posterior_2>posterior_1 and posterior_2>posterior_3:
        list1.append("Iris-virginica")
    elif posterior_3>posterior_1 and posterior_3>posterior_2:
        list1.append("Iris-versicolor")

X_reduce_tested["Predicted"]=list1

# Model's Performance
confusion_m=confusion_matrix(X_reduce_tested["Species"],X_reduce_tested["Predicted"])
model_accuracy_reduced=((confusion_m[0,0]+confusion_m[1,1]+confusion_m[2,2])/np.sum(confusion_m))*100
print("Model's Accuracy of Reduced Data: ",model_accuracy_reduced)


Class_1=file_[file_["Species"]=='Iris-setosa']
Class_2=file_[file_["Species"]=='Iris-virginica']
Class_3=file_[file_["Species"]=='Iris-versicolor']
Class_1=Class_1.drop("Species",axis=1)
Class_2=Class_2.drop("Species",axis=1)
Class_3=Class_3.drop("Species",axis=1)
Covariance_1=np.cov(Class_1,rowvar=False)
Covariance_2=np.cov(Class_2,rowvar=False)
Covariance_3=np.cov(Class_3,rowvar=False)
det_1=np.linalg.det(Covariance_1)
det_2=np.linalg.det(Covariance_2)
det_3=np.linalg.det(Covariance_3)
inverse_cov1=np.linalg.inv(Covariance_1)
inverse_cov2=np.linalg.inv(Covariance_2)
inverse_cov3=np.linalg.inv(Covariance_3)
mean_c1=Class_1.mean()
mean_c2=Class_2.mean()
mean_c3=Class_3.mean()

d=4
list1=[]
list2=[]
list3=[]
for i in range(len(x)):
    Gaussian_p_1=1/(((2*np.pi)**(d/2))*(det_1)**0.5)*np.exp(-0.5*(np.dot(np.dot((x.iloc[i]-mean_c1),inverse_cov1),(x.iloc[i]-mean_c1).T)))
    Gaussian_p_2=1/(((2*np.pi)**(d/2))*(det_2)**0.5)*np.exp(-0.5*(np.dot(np.dot((x.iloc[i]-mean_c2),inverse_cov2),(x.iloc[i]-mean_c2).T)))
    Gaussian_p_3=1/(((2*np.pi)**(d/2))*(det_3)**0.5)*np.exp(-0.5*(np.dot(np.dot((x.iloc[i]-mean_c3),inverse_cov3),(x.iloc[i]-mean_c3).T)))
    posterior_1=(Gaussian_p_1)*(len(Class_1)/len(file_))
    posterior_2=(Gaussian_p_2)*(len(Class_2)/len(file_))
    posterior_3=(Gaussian_p_3)*(len(Class_3)/len(file_))
    list1.append(posterior_1)
    list2.append(posterior_2)
    list3.append(posterior_3)

list_Predicted=[]
for j in range(len(list1)):
    if list1[j]>list2[j] and list1[j]>list3[j]:
        list_Predicted.append("Iris-setosa")
    elif list2[j]>list1[j] and list2[j]>list3[j]:
        list_Predicted.append("Iris-virginica")
    elif list3[j]>list1[j] and list3[j]>list2[j]:
        list_Predicted.append("Iris-versicolor")

file["Predicted"]=list_Predicted

# Model's Performance
confusion_m=confusion_matrix(file["Species"],file["Predicted"])
model_accuracy_original=((confusion_m[0,0]+confusion_m[1,1]+confusion_m[2,2])/np.sum(confusion_m))*100
print("Model's Accuracy of Original Data: ",model_accuracy_original,'\n')

Difference=model_accuracy_original-model_accuracy_reduced
print("The difference between the accuracies of the models built using the original and dimension-reduced data is",Difference)