import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

train=pd.read_csv("abalone_train.csv")
test=pd.read_csv("abalone_test.csv")

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
X=pd.DataFrame(train["Shell weight"])
X["ones"]=1
X=X[["ones","Shell weight"]]
Rings_act=np.array(test["Rings"])
x_test=pd.DataFrame(test["Shell weight"])
x_test["ones"]=1
x_test=x_test[["ones","Shell weight"]]
Ermse_train=[]
Ermse_test=[]
degrees=[2,3,4,5]
for t in range(2,6):
    X[f"Shell weight{t}"]=X["Shell weight"]**t  
    x_test[f"Shell weight{t}"]=x_test["Shell weight"]**t
    w_cap=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Rings_train)
    Rings_pred_test=x_test.dot(w_cap)
    Rings_pred_train=X.dot(w_cap)
    E_rmse_train=(((Rings_pred_train-Rings_train)**2).mean())**0.5
    E_rmse_test=(((Rings_pred_test-Rings_test)**2).mean())**0.5
    print("The prediction accuracy on the test data for p value",t,"using root mean squared error:",round(E_rmse_test,3))
    print("The prediction accuracy on the training data for p value",t,"using root mean squared error:",round(E_rmse_train,3),'\n')
    Ermse_train.append(E_rmse_train)
    Ermse_test.append(E_rmse_test)

best_p=0
for r in range(len(Ermse_train)):
    if Ermse_train[r]==min(Ermse_train):
        best_p=r+2
X[f"Shell weight{best_p}"]=X["Shell weight"]**best_p
w_cap=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Rings_train)
print(w_cap)
x_vals=np.arange(0,1,0.01)
y_vals=np.zeros(len(x_vals))
for i in range(len(w_cap)):
    y_vals+=((x_vals**(i))*w_cap[i])

plt.figure(figsize=(11,7))
plt.scatter(x_train, Rings_train,color='red',alpha=0.5,label='Training Data')
plt.plot(x_vals,y_vals,color="black",label='Best fit (p=5)',alpha=0.5)
plt.xlabel("Shell Weight")
plt.ylabel("Number of Rings")
plt.title("Best-fit Line on Training Data")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.bar(range(2,6),E_rmse_train,label='Training Data',alpha=0.7,color='red')
plt.bar(range(2,6),E_rmse_test,label='Test Data',alpha=0.7,color='black')
plt.xlabel("Degrees of polynomial")
plt.ylabel("RMSE values")
plt.legend()
plt.show()
