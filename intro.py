import streamlit as st

# Imposta il nome che viene fuori nel browser
st.set_page_config(
    page_title="bm25kNN",
    page_icon=":bar_chart:"
)
	
st.title("Analisi dell'effetto k-nn per reperimento di documenti rilevanti")

st.markdown(""" ## Introduzione 

L'information retrieval vuole rispondere a un esigenza informativa, ovvero un insieme di circostanze in cui una persona ha un problema da risolvere che richiede delle informazioni importanti, utili o necessarie per essere risolto.
Questa esigenza \u00e8 vista come uno stato deeficitario chiamato _$\\text{Anomalous State of Knowledge}^1$_
Uno degli aspetti da risolvere \u00e8 il reperimento di documenti rilevanti al problema preso in considerazione, nel nostro caso il problema \u00e8 la risposta a una query. 
Precisiamo che la rilevanza di un documento e la esigenza informativa sono imprescindibili dalla persona in quanto unica a poter dare un giudizio di rilevanza per la sua esigenza. 
 
Per questo progetto dobbiamo anche tenere conto della presenza di informazioni attinenti ma non rilevanti. L'attinenza infatti si riferisce alla relazione di qualcosa con qualcos'altro.
"""
)

st.markdown(""" ### Reperimento
La difficolt\u00e0 nell'IR sta nel dover reperire tutte e solo le informazioni rilevanti e contemporaneamente evitare tutte e solo le informazioni non rilevanti.
Per farlo si usano delle _funzioni di reperimento_ che calcolano un punteggio dai descrittori dei documenti in relazione a quelli della query. Il risultato viene poi usato per ordinare i risultati mettendo in alto i documenti con punteggio maggiore.
Il punteggio nulla ha a che fare con la rilevanza, un documento rilevante si pu\u00f2 trovare al 10 posto e uno non rilevante al 1. 

Per migliorare il reperimento \u00e8 possibile usare diversi metodi, nel nostro caso aggiungeremo dei descrittori alla query iniziale, facendo query expansion (da qui in avanti QE),  derivati dalla vicinanza rispetto altri documenti usando una ricerca dei vicini e un word embedding.         

  
## Obiettivi del Progetto
Questo progetto mostrer\u00e0 le differenze tra il bm25 senza QE e due implementazioni per l'espansione automatica della query: k-NN e Word2Vec; usando un dataset creato dall'uso combinato di Elasticseach, come motore di ricerca, e trec_eval per fare le valutazioni sulle performance del metodo utilizzato.
La collezione di documenti \u00e8 la ROBUST 2004 che comprende circa mezzo milione di documenti di natura giornalstica.
            
Il progetto si suddivide in tre parti:

1. **Descrizione dei metodi usati**: 
         - come funziona il bm25 e il ruolo nel reperimento
         - come funziona QE con k-nn
         - come funziona QE con w2V


2. **Analisi di risultati caratteristici**: vedremo come hanno risposto le query al metodo usato, in confronto al metodo base senza QE.
         
3. **Presentazione dei risultati**: 
         - vedremo se le differenze, qualora presenti, tra i vari metodo proposti siano significative oppure no. 

Le metriche con cui si \u00e8 valutato l'efficacia del reperimento sono le seguenti:            
            
- Precision at 5 (P@5): misura la precisione dei primi 5 documenti restituiti dal motore di ricerca per una determinata query. La precisione, in generale, \u00e8 definita come la frazione di documenti rilevanti tra quelli recuperati; P@5 indica quindi la percentuale di documenti rilevanti tra i primi 5 risultati restituiti.

- MAP (Mean Average Precision): \u00e8 la media della Average Precision (AP) calcolata per ogni query in un dataset. L'Average Precision \u00e8 la media delle precisioni calcolate per ogni posizione in cui \u00e8 stato trovato un documento rilevante. La precisione in un dato punto della lista dei risultati \u00e8 definita come la frazione di documenti pertinenti tra tutti i documenti recuperati fino a quella posizione. L'Average Precision per una query \u00e8 quindi la media della precisione calcolata in ciascuna posizione in cui compare un documento rilevante.

- nDCG@10 (Normalized Discounted Cumulative Gain at 10): \u00e8 una metrica che valuta la qualit\u00e0 del ranking dei primi 10 documenti restituiti. Tiene conto della rilevanza, non binaria, dei documenti e della loro posizione nella lista dei risultati: un documento rilevante in una posizione pi\u00f2 alta contribuisce di pi\u00f2 al punteggio complessivo. La metrica nDCG normalizza il valore ottenuto rispetto al massimo punteggio possibile (DCG ideale), in modo da ottenere un valore compreso tra 0 e 1. Un valore di nDCG@10 pari a 1 indica che tutti i documenti rilevanti si trovano nelle posizioni ottimali nei primi 10 risultati.


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
- (Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013, ottobre)) [Distributed representations of words and phrases and their compositionality.]
- (Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013, gennaio)) [Efficient estimation of word representations in vector space.]
""")