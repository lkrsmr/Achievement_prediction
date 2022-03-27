import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('13term5-01test.xlsx', index_col=0, header=0)

df.dropna(inplace=True)

encoder = LabelEncoder().fit(df["Grade"])

df["Grade"] = encoder.transform(df["Grade"])

df.to_csv('13term5-01test1.csv')
