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
print("Tested Bayes Model:")
print(file,'\n')

# Model's Performance
confusion_m=confusion_matrix(file["Species"],file["Predicted"])
print(confusion_m,'\n')
model_accuracy=((confusion_m[0,0]+confusion_m[1,1]+confusion_m[2,2])/np.sum(confusion_m))*100
print("Model's Accuracy:",model_accuracy)