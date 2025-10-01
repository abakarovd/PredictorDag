from main_folder.DataPipeline import generate_balanced_datasets

generate_balanced_datasets(dag_file="DagReady.csv",
                           ru_file="RuReady.csv",
                           output_dir="/main_folder/DataUP/datafortrainUp1",
                           sample_sizes=(50_000, 100_000, 200_000, 500_000, 1_000_000, 3_000_000, 6_000_000),
                           random_state=42)
