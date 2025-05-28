import pandas as pd
from pathlib import Path

# trova il primo file contenente il pattern
def find_file_by_pattern(folder: Path, keyword: str):
    for file in folder.iterdir():
        if file.is_file() and keyword in file.name:
            return file
    raise FileNotFoundError(f"Nessun file trovato con pattern '{keyword}' in {folder}")

# carica la metrica dato il poath e la metrica a cercare
def load_metric_file(filepath, metric_name):
    df = pd.read_csv(filepath, sep="\t", header=None, names=["metric", "id_Q", metric_name])
    df = df[df["id_Q"] != "all"]
    df["id_Q"] = df["id_Q"].astype(int)
    return df[["id_Q", metric_name]]

#(exp, knn, baseline, w2v)
method_name = "base"

# cartella comune
base_path = Path("D:/Universita/Terzo_Anno/S2/knnw2vdtm_Bm25/Data/EVAL_TEST_Q") / method_name

# trova i file nella directory
map_file = find_file_by_pattern(base_path, "map_map")
p5_file = find_file_by_pattern(base_path, "P5_P5")
ndcg_file = find_file_by_pattern(base_path, "nDCG_nDCG")

# carica le metriche
map_df = load_metric_file(map_file, "map")
p5_df = load_metric_file(p5_file, "p_5")
ndcg_df = load_metric_file(ndcg_file, "ndcg_10")

# merge dei tre dataframe
df_combined = map_df.merge(p5_df, on="id_Q").merge(ndcg_df, on="id_Q")
df_combined["method"] = method_name

# Salva csv.
output_path = base_path / f"Eval_Q_QE_{method_name}_combined.csv"
df_combined.to_csv(output_path, index=False)

# per debug
print(df_combined.head())