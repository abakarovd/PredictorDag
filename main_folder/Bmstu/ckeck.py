import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix

path = "/Users/maybeabakarov/Desktop/PredictorDag/main_folder/ModelsUP/modelsUp4/LogReg095/(2-3)_iter500_C0.15/model/BalancedData_50000.pkl"

# загружаем тестовый файл
test_file = "allName.csv"
df_test = pd.read_csv(test_file)

X_test = df_test["Ф"] + " " + df_test["И"] + " " + df_test["О"]
y_test = df_test["класс"]


print(f"\nМодель: {path}")
model_data = joblib.load(path)
pipeline = model_data["pipeline"]
threshold = model_data["threshold"]

# предсказания
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)


df_to_add = df_test[y_pred == 1].copy()
df_to_add["класс"] = 1  # отмечаем как дагестанцев

df_to_add.to_csv("dagBmstu.csv")