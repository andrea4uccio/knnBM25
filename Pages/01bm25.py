import altair as alt
import polars as pl
import streamlit as st


# Imposta il titolo pagina che viene fuori nel browser
st.set_page_config(
    page_title="bm25kNN"
)


# Importo i dati con la funzione definita in data_cleaning.py (get_data)


st.title("Cosa e\` il bm25 ")
st.write("""
Il BM25 (Best Matching 25) è un algoritmo di ranking usato di default in Elasticsearch per la ricerca di documenti.
Funziona come una misura di rilevanza per determinare quanto un documento sia pertinente rispetto a una query di ricerca.
Il B<M25 calcola un punteggio di rilevanza per ogni documento rispetto alla query, tenendo conto di due metriche principali:
	- Frequenza dei Termini (TF - Term Frequency): Più un termine appare in un documento, maggiore è la sua rilevanza. Relazione non è lineare: l'utilizzo di un termine in modo eccessivo non aumenta indefinitamente il punteggio, ma segue una curva decrescente (saturazione).
	- Frequenza Inversa del Documento (IDF - Inverse Document Frequency): I termini che compaiono in molti documenti sono considerati meno rilevanti. L'IDF riduce il punteggio dei termini che appaiono frequentemente in tutto il corpus.

Questi due fattori vengono combinati in un punteggio finale, che permette al motore di ricerca (nel nostro contesto elasticsearch) di ordinare i documenti in base alla loro rilevanza per la query.

Il BM25 e\` attualmente lo stato dell'arte per la ricerca di documenti rilevanti.  
""")


st.title("Come usiamo il BM25")
st.write("""
In questo progetto il BM25 viene usato da solo, e con QE.
In questa pagina viene trattato il metodo semplice senza QE, vedremo i risultati che proveremo a migliorare succesivamente con QE.  

Per reperire i dati con il BM25 viene usata l'API search() fornita da elasticsearch utilizzando la configurazione base. 
A seguire il codice per il reperimento: 
				
""")

codice = """
...
query_dict = {'match': { '_content': "human genetic code" } }
response = esclient.search(index=indexName, query=query_dict, size=10)
...
"""
st.code(codice, language='python')

bm25 = pl.read_csv("./Data/Eval_Test/test_results_bm25_eval.txt", has_header= False, separator="\t")
dati = bm25.filter((pl.col("column_1").str.starts_with("map")) | (pl.col("column_1").str.starts_with("P_5 ")))

st.write(f"""
I risultati sono poi stati strutturati in modo che il programma trec-eval possa funzionare e fornirci i risultati di rilevanza.
Nel nostro caso usiamo solo map e P_5  

Nel nostro caso abbiamo come risultati:
- map = {dati[0, 2]}
- P_5 = {dati[1, 2]}
""")

st.title("Risultati delle query")

st.text("""
Vediamno ora come le varie query si sono comportate. A seguire verra\` riportato un grafico che mostra l\'andamento di map e P_5 per le query considerate.
				
				""")

query = pl.read_csv("./Data/Eval_Queries/bm25_evalQ.txt", has_header= True)
# Creare il grafico per p5~query
chart_p5 = alt.Chart(query).mark_point().encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=[600, 700])),  # Impostazione dei limiti per l'asse x
    y='p_5'
)

# Visualizzare il grafico
st.altair_chart(chart_p5, use_container_width= True)

st.text(""" 
Da come possiamo vedere ci sono alcune query che hanno un alto valore di P_5 e altre meno.
La Precision at 5 misura la precisione dei primi 5 documenti restituiti dal motore di ricerca per una determinata query.
La precisione, in generale, è definita come la frazione dei documenti rilevanti tra i documenti recuperati, è la percentuale di documenti rilevanti tra i primi 5 risultati restituiti.
Il grafico di per se` non ci racconta molto da solo, stiamo solo vedendo i risultati per il BM25 senza fare confronti.
""")

# Creare il grafico per map~query
chart_map = alt.Chart(query).mark_point().encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=[600, 700])),  # Impostazione dei limiti per l'asse x
    y='map'
)

# Visualizzare il grafico
st.altair_chart(chart_map, use_container_width= True)


st.text(""" 
Stessi risultati di P_5, alcune query performano meglio di altre. 
La MAP è la media dell'Average Precision (AP) calcolata per ogni query in un dataset. 
L'Average Precision è la media della precisione calcolata per ogni posizione in cui è stato trovato un documento rilevante.
La Precision in un dato punto della lista dei risultati è definita come la frazione di documenti pertinenti tra tutti i documenti recuperati fino a quella posizione.
La Average Precision per una query è quindi la media della precisione calcolata per ogni documento rilevante che appare nella lista dei risultati restituiti dal motore di ricerca.
""")


