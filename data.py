import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# keep only required columns
df = df[['tenure','MonthlyCharges','TotalCharges','Churn']]

# convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# remove missing values
df = df.dropna(subset=['TotalCharges'])

# check data
print(df.head())
print(df.shape)

#  Graph 1: Churn count
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Graph 2: Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes":1, "No":0})

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# evaluate
from sklearn.metrics import accuracy_score
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# save model
import pickle
pickle.dump(model, open("model.pkl", "wb"))