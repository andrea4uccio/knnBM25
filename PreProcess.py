import polars as pl

# Funzione per rendere i dati tidy
def clean(filename):
  dati = pl.read_csv(filename, separator="\t")
   # Filtrare i dati per 'metric_name' che è 'map' o 'P_5'
  dati = dati.filter((pl.col("metric_name").str.starts_with("map"))| (pl.col("metric_name").str.starts_with("P_5 ")))
    
  # Separare i dati in map e p_5 utilizzando gli indici alternati
  map_data = dati[::2]  # Ogni seconda riga, iniziando dalla prima
  p_5_data = dati[1::2]  # Ogni seconda riga, iniziando dalla seconda
    
  # Creare un nuovo dataframe con le colonne desiderate
  result = pl.DataFrame({
      "id_Q": p_5_data[:, 1],  # La seconda colonna di p_5
      "p_5": p_5_data[:, 2],   # La terza colonna di p_5
      "map": map_data[:, 2]    # La terza colonna di map
  })
    
  return result

# Scriviamo il nuovo dataset pulito in un file cosi da averlo sempre pronto

tidy = clean("Data/Eval_Queries/test_results_10_25_07_evalQ.txt")
tidy.write_csv("Data/Eval_Queries/10_25_07_evalQ.txt")

tidy = clean("Data/Eval_Queries/test_results_10_100_09_evalQ.txt")
tidy.write_csv("Data/Eval_Queries/10_100_09_evalQ.txt")

tidy = clean("Data/Eval_Queries/test_results_bm25_evalQ.txt")
tidy.write_csv("Data/Eval_Queries/bm25_evalQ.txt")