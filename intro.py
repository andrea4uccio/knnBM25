import streamlit as st

# Imposta il nome che viene fuori nel browser
st.set_page_config(
    page_title="bm25kNN",
    page_icon=":bar_chart:"
)
	
st.title("Analisi dell'effetto k-nn per reperimento di documenti rilevanti")

st.markdown(""" ## Introduzione 

L'information retrieval vuole rispondere a un esigenza informativa, ovvero un insieme di circostanze incui una persona ha un problema da risolvere che richiede delle informazioni importanti, utili o necessarie per essere risolto.
Questa esigenza \u00e8 vista come uno stato deeficitario chiamato _$\\text{Anomalous State of Knowledge}^1$_
Uno degli aspetti da risolvere \u00e8 il reperimento di documenti rilevanti al problema preso in considerazione, nel nostro caso il problema \u00e8 la risposta a una query. 
Precisiamo che la rilevanza di un documento e la esigenza informativa sono imprescindibili dalla persona in quanto unica a poter dare un giudizio di rilevanza per la sua esigenza. 
 
Per questo progetto dobbiamo anche fare tenenere conto della presenza di infomrazioni attinenti ma non rilevanti. L'attinenza infatti si riferisce alla relazione di qualcosa con qualcos'altro, cosa che con l'algoritmo di clustering viene usata.
"""
)

st.markdown(""" ### Reperimento
La difficolt\u00e0 nell'IR sta nel dover reperire tutte e solo le informazioni rilevanti e contemporaneamente evitare tutte e solo le informazioni non rilevanti.
Per farlo si usano delle funzioni chiamate _funzioni di reperimento_ che calcolano un punteggio dai descrittori dei documenti in relazione a quelli della query. Il risultato viene poi usato per ordinare i risultati mettendo in alto i documenti con punteggio maggiore.
Il punteggio nulla ha a che fare con la rilevanza, un documento rilevante si pu\u00f2 trovare al 10 posto e uno non rilevante al 1. 

Per migliorare il reperimento \u00e8 possibile usare diversi metodi, nel nostro caso aggiungeremo dei descrittori alla query iniziale, facendo quindi query expansion (da qui in avanti QE),  derivati dalla vicinanza rispetto altri documenti usando k-nn.         

  
## Obiettivi del Progetto
Questo progetto mostrer\u00e0 le differenze tra il bm25 senza QE e un possibile uso e implementazione del metodo di clustering k nearest neighbours con QE, usando un dataset creato usando Elasticseach come motore di ricerca e trec_eval per fare le valutazioni sulle performance del metodo utilizzato.
         
Il progetto si suddivide in tre parti:

1. **Descrizione dei metodi usati**: 
         - come funziona il bm25 e il ruolo nel reperimento
         - come funziona QE con k-nn

2. **Analisi di risultati caratteristici**: vedremo come hanno risposto le query al metood usato, in confronto al metodo base senza QE.
         
3. **Presentazione dei risultati**: 
         - vedremo se le differenze, qualora presenti, tra i vari metodo proposti siano significative oppure no. 

La Precision at 5 misura la precisione dei primi 5 documenti restituiti dal motore di ricerca per una determinata query.
La precisione, in generale, \u00e8 definita come la frazione dei documenti rilevanti tra i documenti recuperati, \u00e8 la percentuale di documenti rilevanti tra i primi 5 risultati restituiti.
            
La MAP \u00e8 la media dell'Average Precision (AP) calcolata per ogni query in un dataset. 
L'Average Precision \u00e8 la media della precisione calcolata per ogni posizione in cui \u00e8 stato trovato un documento rilevante.
La Precision in un dato punto della lista dei risultati \u00e8 definita come la frazione di documenti pertinenti tra tutti i documenti recuperati fino a quella posizione.
La Average Precision per una query \u00e8 quindi la media della precisione calcolata per ogni documento rilevante che appare nella lista dei risultati restituiti dal motore di ricerca.
            
## Importanza del Progetto

I motori di ricerca non sono importanti solo per trovare pagine internet ma servono anche a molto altro: 
            - ricerca delle milgiori risorse informative (resource finding);
            - individuare una pagina web senza conoscere l'URL (homepage finding);
            - risposta a domande (question answering);
            - ricerca di prevetti industriali (patent search).
Questi sono alcuni dei campi per cui vengono usati i motori di ricerca
L'importanza di trovare e portare in alto i documenti rilevanti \u00e8 fondamentale.
            

## Fonti Utilizzate:
- (Lee, K. S., Croft, W. B., & Allan, J. (2008, July)). [_A cluster-based resampling method for pseudo-relevance feedback._]
- (Smucker, M. D., Allan, J., & Carterette, B. (2007, November) [_A comparison of statistical significance tests for information retrieval evaluation._]
- (Massimo Melucci) [_Information Retrieval Macchine e Motori di ricerca_]
""")