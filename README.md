# K-NN e QE
## Descrizione progetto

Questo progetto ha il compito di esplorare come la query expansion (QE) con k-nn influienzi i risultati delle query rispetto all'assenza di QE.

## Dataset 

I dataset sono stati ottenuti mandando in esecuzione un server ElasticSearch e implementando il reperimento usando k-nn con python (contenuto un estratto nel file `2K_nn.py`) 
A seguito di una fase di training e test si sono scelti i parametri piu adatti e si \u00e8 effettuato il reperimento per poi analizzare i risultati con `trec_eval`.

I dataset sono quindi molteplici e fanno riferimento ai diversi metodo e parametri scelti.

Tutti i dataset contengono molteplici atributi di rilevanza ma per lo studio si valutano le variabili; 
- `map`: mean average precision 
- `P_5`: precision at 5.
Metriche che vengono spiegate nel progetto

## Preprocessing

Il file `preprocess.py` contiene la funzione `getRifJunk()` che si occupa di:
1. Ottenere le sole variabili considerate ( le sopracitate `map` e `P_5`);
2. Rendere i dataset inerenti alle singole query tidy ed eseguire 1).
3. Aggiungere una colonna per il metodo utilizzato e la configurzione.
4. Creare dei file csv puliti che verrano usati per le analisi.

I dati preprocessati vengono poi caricati usando pl.read_csv()

## Librerie usata

- altair : per grafici;
- polars : per preprocessing e manipolazione dati;
- streamlit : per la creazione della interfaccia web;
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

## Analisi
Nella pagina `intro.py` viene fornita una introduzione al problema e alcuni riferimenti usati.
