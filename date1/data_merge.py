import pandas as pd

file1 = pd.read_excel(r'C:\Users\lkrsmr\Desktop\datatest\new5\14term7.xlsx')
file2 = pd.read_excel(r'C:\Users\lkrsmr\Desktop\datatest\new5\termachievement.xlsx')

file3 = file1.merge(file2, on="ID", how="outer")

file3.to_excel(r'C:\Users\lkrsmr\Desktop\datatest\new5\14term7-.xlsx')