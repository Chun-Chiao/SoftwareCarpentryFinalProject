import pandas as pd 
data = pd.read_csv("/Users/chunchiaoyang/Desktop/SoftwareCarpentry/FinalProject/dataset.csv")
print(data.head())
print(data.columns.tolist())