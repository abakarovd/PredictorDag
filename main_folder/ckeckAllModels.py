import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_curve, f1_score
import os
import numpy as np

path = "/main_folder/ModelsUP/models/LogReg09/(2-3)_iter500_C0.15/model/BalancedData_100000.pkl"



def load_all_pipelines_in_order(base_folder):
    """
    Загружает все модели в порядке времени сохранения (от старых к новым).
    Возвращает список кортежей: (полный путь, pipeline, threshold).
    """
    pipelines = []
    file_paths = []

    # собираем все pkl
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    # сортируем по времени изменения файла
    file_paths.sort(key=os.path.getmtime)

    # загружаем по порядку
    for file_path in file_paths:
        data = joblib.load(file_path)
        pipelines.append((file_path, data["pipeline"], data["threshold"]))

    return pipelines

def evaluate_models_on_test(test_file, pipelines, model_v = "modelsName", csv_name="all_models_metrics.csv", plot_name="all_models_pr_curve.png"):
    """
    Проверяет все модели на тестовом файле.
    Сохраняет CSV с метриками и одну общую PR-кривую.
    """

    # Загружаем тестовые данные
    df = pd.read_csv(test_file).fillna("")

    df["full_name"] = df["Ф"] + " " + df["И"] + " " + df["О"]
    X_test = df["full_name"]
    y_test = df["класс"]

    y_test = np.array(y_test).astype(int)

    results = []

    # папка для результатов
    output_folder = "TestData/data"
    os.makedirs(output_folder, exist_ok=True)

    # один общий график
    plt.figure(figsize=(8, 6))

    for path, pipeline, threshold in pipelines:
        model_name = os.path.splitext(os.path.basename(path))[0]

        # вероятность
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        # бинарные предсказания с учетом порога
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

        # имя файла
        ModelName = path.split("/")[-1].split("_")[-1]
        results.append({
            f"{model_v}": ModelName,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "f1": round(f1, 4),
            "Confusion Matrix": cm.tolist()
        })

        print(f"✅ {ModelName}: Precision={precision:.4f}, Recall={recall:.4f}, Threshold={threshold:.4f}")

        # PR-кривая для этой модели
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(rec, prec, label=f"{model_name}")

    # оформление общего графика
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curves - All Models")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # сохраняем общий график и таблицу в TestData/data
    plot_path = os.path.join(output_folder, plot_name)
    csv_path = os.path.join(output_folder, csv_name)

    plt.savefig(plot_path)
    plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)

    print(f"\nТаблица сохранена: {csv_path}")
    print(f"Общая PR-кривая сохранена: {plot_path}")
    return results_df

p = "ModelsPrecisionUP/modelsPrecisionUp"
r = "ModelsUP/modelsUp"
v = 4
p1 = p.split("/")[-1]
r1 = r.split("/")[-1]
pipelines = load_all_pipelines_in_order(f"{r}{v}")

results = evaluate_models_on_test(
    test_file="TestData/filtered_data1.csv",
    pipelines=pipelines,
    model_v=f"{r1}{v}",
    csv_name=f"{r1}{v}.csv",
    plot_name=f"{r1}{v}.png"
)
print(results)
