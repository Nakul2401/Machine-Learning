import pandas as pd 
import numpy as np
Corrected=pd.read_csv("Outliers_Corrected.csv")
#a)
def normalization(df,up,low):
    new_df=df.copy()
    for c in (df.columns[3:]):
        max_xi=np.max(df[c])
        min_xi=np.min(df[c])
        for d in range(len(Corrected[c])):
            normalised_x= ((((df.at[d,c]-min_xi)/(max_xi-min_xi))*(up-low))+low)
            new_df.at[d,c]=normalised_x
    return new_df
normalized_df=normalization(Corrected,12,5)
print(normalized_df.head())

print("\nBefore Normalization:")
for c in (Corrected.columns[3:]):
    min=np.min(Corrected[c])
    max=np.max(Corrected[c])
    print(f'For {c} Minimum={round(min,2)}, Maximum={round(max,2)}')
print("\nAfter Normalization:")
for c in (normalized_df.columns[3:]):
    min=np.min(normalized_df[c])
    max=np.max(normalized_df[c])
    print(f'For {c} Minimum={min}, Maximum={max}')

#b)
def standardization(df):
    new_df=df.copy()
    for c in (df.columns[3:]):
        mean=np.mean(df[c])
        std=np.std(df[c])
        for d in range(len(Corrected[c])):
            standardised_x= (df.at[d,c]-mean)/std
            new_df.at[d,c]=standardised_x
    return new_df
standardized_df=standardization(Corrected)
print(standardized_df.head())

print("\nBefore Standardization:")
for c in (Corrected.columns[3:]):
    mean=np.mean(Corrected[c])
    std=np.std(Corrected[c])
    print(f'For {c} Mean={round(mean,2)}, STD={round(std,2)}')

print("\nAfter Standardization:")
for c in (Corrected.columns[3:]):
    mean=np.mean(standardized_df[c])
    std=np.std(standardized_df[c])
    print(f'For {c} Mean={round(mean,2)}, STD={round(std,2)}')




            
