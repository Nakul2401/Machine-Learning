import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = pd.read_csv('landslide_data_original.csv')

#1)
mean= sum(file['temperature'])/len(file['temperature'])
print("The statistical measures of Temperature attribute are:")
print("Mean=" "{:.2f}".format(mean))

min = min(file['temperature'])
max = max(file['temperature'])
print("Minimum=" "{:.2f}".format(min))
print("Maximum=" "{:.2f}".format(max))

temp = sorted(file['temperature'])
n=len(temp)-1
if n%2==0:
    median = (temp[n//2+1]+temp[n//2])/2
else:
    median = temp[n+1//2]
print("Median=" "{:.2f}".format(median))

list1=[]
for i in temp:
    difference = (i-mean)**2
    list1.append(difference)
standard_deviation = (sum(list1)/len(temp))**0.5
print("STD=" "{:.2f}".format(standard_deviation))

#2)
r=np.zeros((7,7))

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

temperature=list(file['temperature'])
humidity=list(file['humidity'])
pressure=list(file['pressure'])
rain=list(file['rain'])
lightavg=list(file['lightavg'])
lightmax=list(file['lightmax'])
moisture=list(file['moisture'])

attribute=[temperature,humidity,pressure,rain,lightavg,lightmax,moisture]
attributeL=['temperature','humidity','pressure','rain','lightavg','lightmax','moisture']

for i in range(7):
    for j in range (7):
        tr=corr(attribute[i],attribute[j])
        r[i][j]=round(tr,2)
print('\n')

cDF=pd.DataFrame(r,index=attributeL,columns=attributeL)
print(cDF)
print("\n")
print("The redundant attribute with respect to lightavg is lightmax.")
print("\n")

#3)

x=file.groupby('stationid').get_group('t12')
plt.hist(x['humidity'],bins=5)
plt.xlabel('Humidity')
plt.ylabel('Frequency')
plt.title('Humidity for stationid=t12')
plt.show()


    










