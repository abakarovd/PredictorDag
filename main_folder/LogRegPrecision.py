import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
    roc_curve
)

def train_models(
        data_files,
        precision_thresholds=[0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97],  # список нужных порогов по precision
        base_folder="modelsPrecision",
        datafortrain="datafortrain",
        ngram_range=(2, 3),
        max_iter=2000,
        C=1.0,
):
    """
    Обучение логистической регрессии на нескольких CSV-файлах.
    Для каждого precision_thr ищется лучший порог по Recall при Precision >= precision_thr.
    Все модели сохраняются с суффиксом _Pxxx.
    """

    # Папка для текущей версии модели
    base_folder = "ModelsPrecisionUP/" + base_folder
    params_str = str(ngram_range).replace(", ", "-") + f"_iter{max_iter}_C{C}"
    BASE_FOLDER = os.path.join(base_folder, f"LogRegMulti.py/{params_str}")
    DATA_FOLDER = os.path.join(BASE_FOLDER, "data")
    PLOTS_FOLDER = os.path.join(BASE_FOLDER, "plots")
    MODEL_FOLDER = os.path.join(BASE_FOLDER, "model")

    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    results = []
    pr_data = []
    roc_data = []
    file_names = []

    for file in data_files:
        print(f"\nРаботаем с файлом: {file}")
        df = pd.read_csv(f"{datafortrain}/{file}")
        df = df.fillna('')
        df["full_name"] = df["Ф"] + " " + df["И"] + " " + df["О"]
        labels = df["класс"]

        X_train, X_test, y_train, y_test = train_test_split(
            df["full_name"], labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Векторизация
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            solver="sag",
            n_jobs=-1,
            verbose=1,
            C=C
        )
        pipeline = make_pipeline(vectorizer, model)

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = round(time.time() - start_time, 2) / 60  # минуты

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

        # Для каждого precision_thr ищем порог
        for p_thr in precision_thresholds:
            best_threshold = 0.5
            best_recall = 0
            best_precision = 0

            for p, r, t in zip(precision, recall, np.append(thresholds, 1.0)):
                if p >= p_thr and r > best_recall:
                    best_recall = r
                    best_precision = p
                    best_threshold = t

            y_pred_best = (y_proba >= best_threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred_best, labels=[1, 0])

            metrics = {
                "File": file,
                "Precision_thr": p_thr,
                "Best Threshold": best_threshold,
                "Precision": precision_score(y_test, y_pred_best),
                "Recall": recall_score(y_test, y_pred_best),
                "F1": f1_score(y_test, y_pred_best),
                "Confusion Matrix": cm,
                "Training Time (m)": training_time
            }
            results.append(metrics)

            model_name = file.replace(".csv", "") + f"_P{str(p_thr).replace('0.', '')}"
            file_names.append(model_name)

            print(f"\n[{file}] Precision≥{p_thr}: "
                  f"порог={best_threshold:.2f}, P={best_precision:.3f}, R={best_recall:.3f}")

            # Сохраняем модель
            joblib.dump({"pipeline": pipeline, "threshold": best_threshold},
                        f"{MODEL_FOLDER}/{model_name}.pkl")

            # Данные для общих PR и ROC
            pr_data.append((recall, precision, recall_score(y_test, y_pred_best), best_precision))
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            roc_data.append((fpr, tpr, auc_score))

    # CSV с метриками
    results_df = pd.DataFrame(results)
    numeric_cols = ["Best Threshold", "Precision", "Recall", "F1", "Training Time (m)"]
    results_df[numeric_cols] = results_df[numeric_cols].round(3)

    summary_path = os.path.join(BASE_FOLDER, f"summary_metrics.csv")
    results_df.to_csv(summary_path, index=False)

    print("\n=== Сравнение всех файлов и всех Precision_thr ===")
    print(results_df)
    print(f"\nФайл с метриками сохранен: {summary_path}")

    # Общая PR кривая
    plt.figure(figsize=(8, 6))
    for name, (recall_vals, precision_vals, r_best, p_best) in zip(file_names, pr_data):
        plt.plot(recall_vals, precision_vals, label=f"{name}")
        plt.scatter([r_best], [p_best], color='red')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall всех выборок")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_FOLDER}/precision_recall_all.png")
    plt.close()


data_files = [

    "BalancedData_50000.csv",
    "BalancedData_100000.csv",
    "BalancedData_200000.csv",
    "BalancedData_500000.csv",
    "BalancedData_1000000.csv",
    "BalancedData_3000000.csv",
    "BalancedData_6000000.csv",

]
train_models(data_files,
             base_folder="modelsPrecisionUp4",
             datafortrain="DataUp/datafortrainUp4",
             ngram_range=(2, 3),
             max_iter=500,
             C=0.15)


