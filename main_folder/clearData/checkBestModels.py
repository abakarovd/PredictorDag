import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix

# список моделей
model_paths = [
    "",
]

# загружаем тестовый файл
test_file = "RuReady.csv"
df_test = pd.read_csv(test_file)

X_test = df_test["Ф"] + " " + df_test["И"] + " " + df_test["О"]
y_test = df_test["класс"]

conf_matrices = []
preds = []  # сюда складываем предсказания моделей

# прогоняем все модели
for path in model_paths:
    print(f"\nМодель: {path}")
    model_data = joblib.load(path)
    pipeline = model_data["pipeline"]
    threshold = model_data["threshold"]

    # предсказания
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    preds.append(y_pred)



# Добавляем строки модели в DagReady.csv (и меняем класс на 1)
df_to_add = df_test[preds[0] == 1].copy()
df_to_add["класс"] = 1  # переотмечаем как дагестанцев
dag_file = "DagReady.csv"
if os.path.exists(dag_file):
    df_dag = pd.read_csv(dag_file)
else:
    df_dag = pd.DataFrame(columns=["Ф", "И", "О", "класс"])
df_dag = pd.concat([df_dag, df_to_add], ignore_index=True)
df_dag.to_csv(dag_file, index=False)
print(f"Добавлено {len(df_to_add)} строк в {dag_file} (класс переотмечен на 1)")

# Удаляем строки модели из RuReady.csv
df_updated = df_test[preds[0] != 1].copy()
df_updated.to_csv(test_file, index=False)
print(f"Удалено {len(df_test) - len(df_updated)} строк из {test_file}")

