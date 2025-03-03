import streamlit as st

# Imposta il nome che viene fuori nel browser
st.set_page_config(
    page_title="bm25kNN",
    page_icon=":bar_chart:"
)
	
st.title("Analisi dell'effetto k-nn per reperimento di documenti rilevanti")

st.write("""

In Information retrieval uno degli aspetti piu\` ricercati sono i metodi per reperire il maggior numero di pagine rilevanti a una richiesta e che queste vengano posizionate all'inizio dei risultati restituti.

La funzione di riperimento confronta i descrittori presenti nella query con i descrittori dei documenti. A ogni documento viene poi assegnato un punteggio in base al peso dei singoli descrittori. 
Importante considerare che un documento con alto punteggio non garantisce che sia rilevante e viceversa. 

Per migliorare il reperimento e\` possibile usare diversi metodi, nel nostro caso aggiungeremo dei descrittori alla query iniziale, facendo quindi query expansion (da qui in avanti QE),  derivati dalla vicinanza rispetto altri documenti usando k-nn.         

### Obiettivi del Progetto
Questo progetto mostrera\` le differenze tra il bm25 senza QE e un possibile uso e implementazione del metodo di clustering k nearest neighbours con QE, usando un dataset creato usando Elasticseach come motore di ricerca e trec_eval per fare le valutazioni sulle performance del metodo utilizzato.
         
Il progetto si suddivide in tre parti:

1. **Descrizione dei metodi usati**: 
         - come funziona il bm25 e il ruolo nel reperimento
         - come funziona QE con k-nn

2. **Impolementazione e creazione del dataset**: parleremo di come sono stati scelti i parametri e di come i documenti sono stati suddivisi per fare training e testing
         
3. **Presentazione dei risultati**: Indagheremo come i diversi parametri influenzano sulle variabili studiate, almeno marginalmente;
         - le query che subiscono la maggiore variazione iin precisione

Per visitare le varie sezioni, aprire il menù sulla sinistra.

### Importanza del Progetto

I motori di ricerca non sono importanti solo per trovare pagine internet ma servono anche a trovare brevetti, e tanto altro...
L'importanza di trovare e portare in alto i documenti rilevanti e\` alla base di IR, e se il metodo non e\` particolarmente computazionalmente oneroso implementarlo in una motore di ricerca che migliora i risultati fa solo che bene.

**Fonti Utilizzate**:
- (Lee, K. S., Croft, W. B., & Allan, J. (2008, July)). [_A cluster-based resampling method for pseudo-relevance feedback._]
- (Smucker, M. D., Allan, J., & Carterette, B. (2007, November) [_A comparison of statistical significance tests for information retrieval evaluation._]
- (Massimo Melucci) [_Information Retrieval Macchine e Motori di ricerca_]
""")