import pandas as pd

df = pd.read_csv('data.csv')
relevance = df.corr()

# print(relevance)
relevance.to_csv('result.csv')

