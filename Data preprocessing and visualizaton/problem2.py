import pandas as pd
import matplotlib.pyplot as plt
#1)
file= pd.read_csv("landslide_data_miss.csv")
original=pd.read_csv("landslide_data_original.csv")
original["dates"]=pd.to_datetime(file["dates"],dayfirst=True)
file= file.dropna(subset='stationid')
file["dates"]=pd.to_datetime(file["dates"],dayfirst=True)
file= file.drop(file[file.isna().sum(axis=1)>2].index)
na_counts=file.isna().sum()
#2)
file.reset_index(inplace=True,drop=True)
for i in (file.columns[2:]):
    for j in range(len(file)):
        if pd.isna(file.loc[j,i]):
            x=file.loc[j,file.columns[0]]
            for k in range(j-1,-1,-1):
                if  pd.isna(file.loc[k,i]):
                    continue
                else:
                    y1=file.loc[k,i]
                    x1=file.loc[k,file.columns[0]]
                    break    
            
            for k in range(j+1,len(file)):
                if  pd.isna(file.loc[k,i]):
                    continue
                else:
                    y2=file.loc[k,i]
                    x2=file.loc[k,file.columns[0]]
                    break
            file.loc[j,i]=(((y2-y1)*((x-x1)/(x2-x1))))+y1

file.to_csv("landslide_data_linearlyInterpolated.csv")
print(file)
interpolated=pd.read_csv("landslide_data_linearlyInterpolated.csv")
#a)
for c in (interpolated.columns[3:]):
    mean=sum(original[c])/len(original[c])
    mean2= sum(interpolated[c])/len(interpolated[c])
    print(f"The statistical measures of {c} attribute are:")
    print("Mean(original)=" "{:.2f}".format(mean)+"    Mean(After Interpolation)=" "{:.2f}".format(mean2))

    sorted_c = sorted(interpolated[c])
    n=len(sorted_c)-1
    sorted_o=sorted(original[c])
    m=len(sorted_o)-1
    if m%2==0:
        median = (sorted_o[m//2+1]+sorted_o[m//2])/2
    else:
        median2= sorted_o[m+1//2]
    if n%2==0:
        median2 = (sorted_c[n//2+1]+sorted_c[n//2])/2
    else:
        median2 = sorted_c[n+1//2]
    print("Median(original)=" "{:.2f}".format(median)+"  Median(After Interpolation)=" "{:.2f}".format(median2))
    list_i=[]
    list_o=[]
    for i in sorted_o:
        difference = (i-mean)**2
        list_o.append(difference)
    for j in sorted_c:
        difference = (j-mean2)**2
        list_i.append(difference)
    standard_deviation = (sum(list_o)/len(sorted_o))**0.5
    standard_deviation2 = (sum(list_i)/len(sorted_c))**0.5
    print("STD(original)=" "{:.2f}".format(standard_deviation)+"     STD(After Interpolation)=" "{:.2f}".format(standard_deviation2))
    print("\n")
#b)
rmse=[]
for l in (file.columns[2:]):
    numerator=0
    for n in range(len(interpolated)):
        numerator+=(interpolated.at[n,l]-original[(original["dates"]==interpolated.at[n,"dates"]) & (original["stationid"]==interpolated.at[n,"stationid"])][l].values[0])**2
    rmse.append((numerator/na_counts[l])**0.5)
print("RMSEs: ",rmse)

attributeL=['temperature','humidity','pressure','rain','lightavg','lightmax','moisture']
plt.plot(attributeL,rmse,color="red",marker='s',markersize=4)
plt.grid(color='blue',linestyle='--',linewidth=0.5)
plt.xlabel("Attribute")
plt.ylabel("RMSEs")
plt.title("RMSEs vs Attributes")
plt.show()

















