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
    confusion_matrix, precision_recall_curve,
    roc_curve
)

def train_models(
        data_files,
        recall_thr = 0.9,
        base_folder="models",
        datafortrain= "datafortrain",
        ngram_range=(2, 3),
        max_iter=2000,
        C=1.0,
):

    """
    Обучение логистической регрессии на нескольких CSV-файлах.
    Сохраняет модели и пороги, метрики, PR/ROC кривые, графики топ-признаков.
    Все модели будут храниться в base_folder
    Данные для обучения будут браться из директории datafortrain
    Фиксация Recall через recall_thr
    """

    # Папка для текущей версии модели
    base_folder = "ModelsUP/" + base_folder
    params_str = str(ngram_range).replace(", ", "-") + f"_iter{max_iter}_C{C}"
    a = str(recall_thr).replace(".", "")
    BASE_FOLDER = os.path.join(base_folder, f"LogReg{(a)}/{params_str}")
    DATA_FOLDER = os.path.join(BASE_FOLDER, "data")
    PLOTS_FOLDER = os.path.join(BASE_FOLDER, "plots")
    MODEL_FOLDER = os.path.join(BASE_FOLDER, "model")

    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    results = []
    pr_data = []
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
        """
        class_weight="balanced" - автоматически балансирует веса классов, если у классов не поровну 
        max_iter=max_iter - максимальное количество итераций для сходимости оптимизатора.
        solver="sag" - метод оптимизации (алгоритм обучения).
        sag (Stochastic Average Gradient) — быстрый стохастический градиент, хорошо работает на больших разреженных матрицах, именно таких, какие даёт TF-IDF.
        n_jobs=-1 - использовать все доступные ядра процессора для параллельных вычислений
        verbose=1 - выводит в консоль прогресс обучения 
        C=C - параметр регуляризации. 
        """

        pipeline = make_pipeline(vectorizer, model)

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = round(time.time() - start_time, 2) / 60  # минуты

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

        # Лучший порог для Recall
        best_threshold = 0.5
        best_precision = 0
        for p, r, t in zip(precision, recall, np.append(thresholds, 1.0)):
            if r >= recall_thr and p > best_precision:
                best_precision = p
                best_threshold = t

        y_pred_best = (y_proba >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_best,labels= [1, 0])
        feature_names = vectorizer.get_feature_names_out()
        metrics = {
            "File": file,
            "Best Threshold": best_threshold,
            "Precision": precision_score(y_test, y_pred_best),
            "Recall": recall_score(y_test, y_pred_best),
            "F1": f1_score(y_test, y_pred_best),
            "Confusion Matrix": cm,
            "Feature": len(feature_names),
            "Training Time (m)": training_time
        }
        results.append(metrics)
        file_names.append(file.replace(".csv", ""))

        print(f"\nЛучший порог: {best_threshold:.2f}, Precision: {best_precision:.4f}, Recall {recall_thr}")
        print("Метрики на тесте:")
        print(metrics)
        print("Confusion Matrix:")
        print(cm)

        # Сохраняем модель
        model_name = file.replace(".csv", "")
        joblib.dump({"pipeline": pipeline, "threshold": best_threshold},
                    f"{MODEL_FOLDER}/{model_name}.pkl")

        #Сохраняем коэффициенты признаков
        coefficients = model.coef_[0]
        feat_df = pd.DataFrame({'feature': feature_names, 'coef': coefficients})
        feat_df['impact'] = feat_df['coef'].apply(lambda x: 'positive' if x > 0 else 'negative')
        feat_df_sorted = feat_df.sort_values(by='coef', ascending=False)
        feat_df_sorted.to_csv(f"{DATA_FOLDER}/all_features_sorted_{model_name}.csv", index=False)


        # Топ-30 признаков
        top_positive = feat_df_sorted.head(30)
        top_negative = feat_df_sorted.tail(30)
        features_plot = list(top_negative['feature'][::-1]) + list(top_positive['feature'])
        coefs_plot = list(top_negative['coef'][::-1]) + list(top_positive['coef'])
        colors_plot = ['red'] * 30 + ['green'] * 30

        plt.figure(figsize=(12, 8))
        plt.barh(features_plot, coefs_plot, color=colors_plot)
        plt.xlabel("Коэффициент логрегрессии")
        plt.title(f"Топ-30 паттернов: {file}")
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_FOLDER}/top_features_{model_name}.png")
        plt.close()

        # Данные для общих PR
        pr_data.append((recall, precision, recall_score(y_test, y_pred_best), best_precision))
        fpr, tpr, _ = roc_curve(y_test, y_proba)

    # CSV с метриками
    results_df = pd.DataFrame(results)
    numeric_cols = ["Best Threshold", "Precision", "Recall", "F1", "Feature", "Training Time (m)"]
    results_df[numeric_cols] = results_df[numeric_cols].round(3)

    summary_path = os.path.join(BASE_FOLDER, f"summary_metrics_ngram{ngram_range}_iter{max_iter}_C{C}.csv")
    results_df.to_csv(summary_path, index=False)

    print("\n=== Сравнение всех файлов ===")
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
    #"BalancedData_6000000.csv",

]

train_models(data_files,
             base_folder="modelsUp4",
             datafortrain = "DataUp/datafortrainUp4",
             recall_thr=0.95,
             ngram_range=(2, 3),
             max_iter=500,
             C=0.15)

