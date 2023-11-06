import pandas as pd
import matplotlib.pyplot as plt

interpolated = pd.read_csv("landslide_data_linearlyInterpolated.csv")
def mid(icol):
    sorted_file = sorted(icol)
    n=len(sorted_file)-1
    if n%2==0:
        return (sorted_file[n//2+1]+sorted_file[n//2])/2
    else:
        return sorted_file[n+1//2]
    
#a)
count=1
fig = plt.figure(figsize=(12,8))
description=interpolated.describe()
for i in (interpolated.columns[3:]):
    outliers=[]
    lower_quartile=description[i]['25%']
    upper_quartile=description[i]['75%']
    inter_quartile_range = upper_quartile-lower_quartile
    for j in interpolated[i]:
        if(not (lower_quartile - 1.5*inter_quartile_range < j < upper_quartile +1.5*inter_quartile_range)):
            outliers.append(j)
    print(f'{i}:',outliers)
    ax = fig.add_subplot(3,3,count)
    ax.boxplot(interpolated[i])
    ax.set_title(i)
    count+=1
fig.tight_layout()
plt.show()

#b)
count=1
fig = plt.figure(figsize=(12,8))
for i in (interpolated.columns[3:]):
    outliers=[]
    lower_quartile=description[i]['25%']
    upper_quartile=description[i]['75%']
    inter_quartile_range = upper_quartile-lower_quartile
    for j in range(len(interpolated[i])):
        median=mid(interpolated[i])
        if(not (lower_quartile - 1.5*inter_quartile_range < interpolated.at[j,i] < upper_quartile +1.5*inter_quartile_range)):
            interpolated.at[j,i]=median
            # outliers.append(interpolated.at[j,i])
    # print(f'{i}:',outliers)
    ax = fig.add_subplot(3,3,count)
    ax.boxplot(interpolated[i])
    ax.set_title(i)
    count+=1
fig.tight_layout()
plt.show()
interpolated.to_csv("Outliers_Corrected.csv",index=False)












