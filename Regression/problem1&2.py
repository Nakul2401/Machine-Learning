import pandas as pd
import numpy as np 
import sklearn.model_selection
import matplotlib.pyplot as plt

#1
file=pd.read_csv("abalone.csv")
train,test=sklearn.model_selection.train_test_split(file,random_state=42,test_size=0.30,shuffle=True)
train.to_csv("abalone_train.csv",index=False)
test.to_csv("abalone_test.csv",index=False)

#2
def corr(x,y):
    xm=np.mean(x)
    ym=np.mean(y)
    var1,var2,var3=0,0,0
    for i in range(len(x)):
        var1+=(x[i]-xm)*(y[i]-ym)
        var2+=(x[i]-xm)**2
        var3+=(y[i]-ym)**2
    r= var1/((var2*var3)**0.5)
    return r

Length=list(train['Length'])
Diameter=list(train['Diameter'])
Height=list(train['Height'])
Whole_weight=list(train['Whole weight'])
Shucked_weight=list(train['Shucked weight'])
Viscera_weight=list(train['Viscera weight'])
Shell_weight=list(train['Shell weight'])
Rings=list(train['Rings'])

Shell_weight_test=list(test['Shell weight'])
Rings_test=list(test['Rings'])

attributes=[Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight]
attributes_string=['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
print("\n")
for i in range(7):
    tr=corr(attributes[i],Rings)
    print("The pearson correlation cofficient of",attributes_string[i],"with target attribute is:",tr)

print("\n")
print("As we can see that Shell weight is having highest pearson correlation cofficient with the target attribute Rings ie.",corr(Shell_weight,Rings),"so we'll build simple linear regression model with input variable as Shell weight(independent variable) and output variable(dependent variable) as Rings","\n")


x_train=np.array(Shell_weight)
Rings_train=np.array(train["Rings"])
X=pd.DataFrame(train['Shell weight'])
X["ones"]=1
X=X[["ones","Shell weight"]]
w_cap= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Rings_train)

x_test=pd.DataFrame(test["Shell weight"])
x_test["ones"]=1
x_test=x_test[["ones","Shell weight"]]

Rings_act=np.array(test["Rings"])
Rings_pred_test=x_test.dot(w_cap)
Rings_pred_train=X.dot(w_cap)
print(Rings_test)
E_rmse_train=(((Rings_pred_train-Rings_train)**2).mean())**0.5
E_rmse_test=(((Rings_pred_test-Rings_test)**2).mean())**0.5
print("The prediction accuracy on the test data using root mean squared error:",E_rmse_test)
print("The prediction accuracy on the training data using root mean squared error:",E_rmse_train,'\n')
plt.figure(figsize=(11,7))
plt.scatter(x_train, Rings_train,color='red',alpha=0.5,label='Training Data')
plt.plot(x_train, X.dot(w_cap),color='black',label='Best fit line',alpha=0.5)
plt.xlabel("Chosen Attribute Value")
plt.ylabel("Number of Rings")
plt.title("Best-fit Line on Training Data")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(Rings_act,Rings_pred_test,color='red',alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Scatter plot of actual Rings (x-axis) vs predicted Rings (y-axis) on the test data")
plt.show()



