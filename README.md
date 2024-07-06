# PRODIGY_DS_2-Titanic-Data-Cleaning-And-EDA

import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(url)

print(titanic_df.head())

missing_values = titanic_df.isnull().sum()
print(missing_values)

titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

titanic_df.drop(columns=['Cabin'], inplace=True)

print(titanic_df.describe())

categorical_cols = ['Survived', 'Pclass', 'Sex', 'Embarked']
for col in categorical_cols:
    print(titanic_df[col].value_counts())
    print("\n")
import matplotlib.pyplot as plt

titanic_df.groupby('Sex')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.show()

titanic_df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(10, 6))
titanic_df[titanic_df['Survived'] == 1]['Age'].plot(kind='hist', bins=30, alpha=0.5, color='blue', label='Survived')
titanic_df[titanic_df['Survived'] == 0]['Age'].plot(kind='hist', bins=30, alpha=0.5, color='red', label='Not Survived')
plt.legend()
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
