# K-NN, W2V e QE
## Descrizione progetto

Questo progetto ha il compito di esplorare come la query expansion (QE) con k-nn e Word2Vec influenzi i risultati delle query rispetto a una baseline in assenza di QE.

## Dataset 

I dataset sono stati ottenuti mandando in esecuzione un server ElasticSearch e implementando il reperimento usando k-nn e word2vec con python (contenuto un estratto nel file `2K_nn.py` e `03w2v.py`) 
A seguito di una fase di training e test si sono scelti i parametri piu adatti e si \u00e8 effettuato il reperimento per poi analizzare i risultati con `trec_eval`.

I dataset sono quindi molteplici e fanno riferimento ai diversi metodo e parametri scelti.

Tutti i dataset contengono molteplici atributi di rilevanza ma per lo studio si valutano le variabili; 
- `map`: mean average precision; 
- `P_5`: precision at 5;
- `mDCG_10` : normalized discounted cumulative gain at 10. 
Metriche che vengono spiegate nel progetto

## Preprocessing

Il file `preprocess.py` si occupa di mettere assieme i file contenti nella cartella `EVAL_TEST_Q` ineerenti ai metodi utilizzati e creare ununico file .csv per metodo. Cosi che per ogni metodo il file csv contenga per ogni colonna la metrica relativa alla query(riga);

I dati preprocessati vengono poi caricati usando pl.read_csv()

## Librerie usata

- altair : per grafici;
- polars : per preprocessing e manipolazione dati;
- streamlit : per la creazione della interfaccia web;
- plotly : per la creazione di grafici 3D interattivi;
- scipy : per i test di significativit&#224;.

## Come Usare il Codice

L'utilizzo di **uv** consente di eseguire il codice in modo semplice.

1. **Scaricare la repository:**

   - Clonare la repository sul proprio PC:
     ```bash
     git clone https://github.com/andrea4uccio/knnBM25.git
     ```
2. **Esecuzione:**

   - Avviare l'app Streamlit:
     ```bash
     uv run streamlit run intro.py
     ```
Se non compare il menu\` a tendina a sinistra rinominare la cartella `Pages` in `pages`
## Analisi
Nella pagina `intro.py` viene fornita una introduzione al problema e alcuni riferimenti usati.
