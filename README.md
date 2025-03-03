# pippo 
## Descrizione progetto

Questo progetto ha il compito di esplorare come diverse funzioni di riperimento influsicono sui risultati di un motore di ricerca.
Si sono in particolare usati k-nn e il bm25.

## Dataset 

I dataset sono stati ottenuti mandando in esecuzione un server ElasticSearch e implementando il reperimento usando k-nn con python (contenuto nel file `knn_search.py`) 
A seguito di una fase di training e test is sono scelti i parametri piu adatti e si e\` effettuato il reperimento per poi analizzare i risultati con `trec_eval`.

I dataset sono quindi molteplici e fanno riferimento ai diversi metodo e parametri scelti.

Tutti i dataset contengono molteplici atributi di rilevanza ma per lo studio si valutano le variabili; 
- `map`: mean average precision 
- `P_5`: precision @ 5 che corrisponde a

## Preprocessing

Il file `preprocess.py` contiene la funzione `getRifJunk()` che si occupa di:
1. Ottenere le sole variabili considerate ( le sopracitate `map` e `P_5`);
2. Rendere i dataset inerenti alle singole query tidy ed eseguire 1).

I dati preprocessati vengono poi caricati usando `getData()` contenuta nello stesso file.

## Librerie usata

pending ma sicuro qualcosa per fare i grafici
polars
streamlint
plotly
