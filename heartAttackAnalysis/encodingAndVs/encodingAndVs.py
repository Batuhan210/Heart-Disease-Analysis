import pandas as pd  
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns 


# read file
df = pd.read_csv('heartAttackAnalysis/data/heart.csv')


# Encoding - We converted some data to numerical values.
Label_encoder = LabelEncoder()
df['Sex'] = Label_encoder.fit_transform(df['Sex'])
df['ChestPainType'] = Label_encoder.fit_transform(df['ChestPainType'])
df['RestingECG'] = Label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = Label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = Label_encoder.fit_transform(df['ST_Slope'])


# visualization 
df.hist(figsize=(16,12), bins=20, edgecolor= 'black', color='lightgreen')
plt.suptitle('Data Distribution')
plt.show()


# A chart comparing age and cholesterol levels according to heart disease
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x = 'Age', y = 'Cholesterol', hue='HeartDisease')
plt.title('The Relationship Between Age and Cholesterol')
plt.show()


# Heatmap graph, values ​​of 1 are associated with disease
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heat Map')
plt.show()


# How many people are sick and how many are not sick graph
plt.figure(figsize=(6,6))
sns.countplot(data=df, x='HeartDisease', color='skyblue')
plt.title('Heart Disease Rates')
plt.show()
