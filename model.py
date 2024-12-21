import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("breast cancer.csv")
print(data.head())

data.drop("Unnamed: 32",axis=1,inplace=True)

print(data.diagnosis.sum())

x = data.drop("diagnosis",axis = 1)
y =  data["diagnosis"]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train,y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test,predictions)
print(f"accuracy = {accuracy:.4f}")

results = pd.DataFrame({"Actual value " : y_test, "Predicted Value " : predictions })
print(results)

for i in range(len(y_test)):
    if y_test.iloc[i] != predictions[i]:
        print(f"{i}) {y_test.iloc[i]} , {predictions[i]}" )