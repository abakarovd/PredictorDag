import pandas as pd
from sklearn.utils import shuffle
import os

def clean_dataframe(df: pd.DataFrame, drop_col: str = None) -> pd.DataFrame:
    """
    Очистка датафрейма:
    1. Удаление NaN
    2. Приведение строк к нижнему регистру
    3. Удаление дубликатов
    """
    # Удаляем ненужный столбец
    if drop_col and drop_col in df.columns:
        df = df.drop(drop_col, axis=1)
    # Удаляем NaN
    df = df.dropna()
    # Приводим все строковые столбцы к нижнему регистру
    for col in df.select_dtypes(include="object").columns:
        df.loc[:, col] = df[col].str.lower()
    # Удаляем дубликаты
    df = df.drop_duplicates()
    return df

def prepare_dataset(dag_file="data/DagestanNames.csv",
                    ru_file="data/RussianNames.csv",
                    output_combined="data/DataAboutNames.csv") -> pd.DataFrame:
    """Подготавливает Dagestan и Russian датасеты """
    # Dagestan
    df_dag = pd.read_csv(dag_file)
    df_dag.columns = ["Ф", "И", "О"]
    df_dag["класс"] = 1
    df_dag = clean_dataframe(df_dag)
    df_dag.to_csv("data/DagReady.csv", index=False)

    # Russian
    df_ru = pd.read_csv(ru_file, names=["Ф", "И", "О", "Пол"])
    df_ru["класс"] = 0
    df_ru = clean_dataframe(df_ru, drop_col="Пол")
    df_ru.to_csv("data/RuReady.csv", index=False)

def generate_balanced_datasets(dag_file="data/DagReady.csv",
                               ru_file="data/RuReady.csv",
                               output_dir="DataUp/datafortrain",
                               sample_sizes=(50_000, 100_000, 200_000, 500_000, 1_000_000, 3_000_000, 6_000_000),
                               random_state=42):
    """
    Генерирует несколько сбалансированных датасетов:
    - Берёт всех из dag_file (дагестанцы, класс=1)
    - Подмешивает случайные подвыборки из ru_file (русские, класс=0)
    - Сохраняет результат в CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем файлы
    dag = pd.read_csv(dag_file)   # все дагестанцы (~54k)
    ru = pd.read_csv(ru_file)     # все русские (~6M)

    print(f"[ИНФО] Загружено: {dag.shape[0]} дагестанцев и {ru.shape[0]} русских")

    for size in sample_sizes:
        # Берём подвыборку русских
        ru_sampled = ru.sample(n=size, random_state=random_state)

        # Объединяем с дагестанцами
        df_balanced = pd.concat([dag, ru_sampled], axis=0)
        df_balanced = shuffle(df_balanced, random_state=random_state).reset_index(drop=True)

        # Сохраняем
        output_file = os.path.join(output_dir, f"BalancedData_{size}.csv")
        df_balanced.to_csv(output_file, index=False)

        print(f"[ИНФО] Сбалансированный набор BalancedData_{size}.csv сохранён")
        print(df_balanced["класс"].value_counts())


