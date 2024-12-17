import pandas as pd  
from sklearn.preprocessing import LabelEncoder, StandardScaler


# read the excel file
df = pd.read_csv('heartAttackAnalysis/data/heart.csv')
# # print(df.head())
# print(df.info())
# print(df.isnull().sum())

# Categorical Columns - Convert to numeric values
categorical_columns = ['ChestPainType', 'Sex', 'ExerciseAngina', 'ST_Slope']

# Encoding
le = LabelEncoder()
for column in categorical_columns:
    df[column] = le.fit_transform(df[column])
# print(df.head())


# Make standart
numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print(df.head())

# save file as csv
df.to_csv('edited_heartAttack.csv', index=False)