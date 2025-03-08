import altair as alt
import polars as pl
import streamlit as st

# Imposta il titolo pagina che viene fuori nel browser
st.set_page_config(
    page_title="bm25kNN"
)


st.title("Cosa \u00e8 il bm25 ")
st.markdown("""
Il BM25 (Best Matching 25) \u00e8 un algoritmo di ranking usato di default in Elasticsearch per la ricerca di documenti.
Funziona come una misura di rilevanza per determinare quanto un documento sia pertinente rispetto a una query di ricerca.
Il B<M25 calcola un punteggio di rilevanza per ogni documento rispetto alla query, tenendo conto di due metriche principali:
- Term Frequency (TF): La frequenza di un termine in un documento. BM25 tiene conto che la rilevanza di un termine decresce all'aumentare della sua frequenza (legge di saturazione).
- Inverse Document Frequency (IDF): Misura l'importanza di un termine. Un termine che appare in pochi documenti \u00e8 più rilevante di uno che appare in molti documenti. BM25 utilizza l'IDF per dare più peso ai termini rari.
- Funzione di Saturazione: La funzione di BM25 non cresce linearmente con la frequenza del termine. Anzi, dopo un certo punto, l'aggiunta di ulteriori occorrenze del termine non aumenta significativamente la rilevanza.
- Parametri:
  - k1: Un parametro che regola la sensibilit\u00e0 alla frequenza del termine.
  - b: Un parametro che regola l'importanza della lunghezza del documento. Un valore di b=1 considera i documenti più lunghi come più rilevanti, mentre un valore più basso riduce questo effetto.
Questi due fattori vengono combinati in un punteggio finale, che permette al motore di ricerca (nel nostro contesto elasticsearch) di ordinare i documenti in base alla loro rilevanza per la query.

In quesot progetto i parametri k1 e b non vengono modificati, vengono usati i valori default di ElasticSearch. 
Il BM25 \u00e8 attualmente lo stato dell'arte per la ricerca di documenti rilevanti.  
""")


st.markdown("## Reperimento senza QE")
st.write("""
In questa sezione otteniamo i risultati delle query senza fare QE. Per il reperimento dei documenti viene usata l'API Search(), che tramite il BM25 restituisce una lista ordinata di documenti come precedentemente detto.
I risultati poi ottenuti saranno quelli da "battere' con il metodo proposto del k-nn.
A seguire il codice per il reperimento
				
""")

codice = """
...
query_dict = {'match': { '_content': "human genetic code" } }
response = esclient.search(index=indexName, query=query_dict, size=10)
...
"""
st.code(codice, language='python')

#Recupero dati
bm25 = pl.read_csv("./Data/Eval_Test/test_results_bm25_eval.txt", has_header= False, separator="\t")
#Filtro le righe per selezionare solo i risultati map e P@5
dati = bm25.filter((pl.col("column_1").str.starts_with("map")) | (pl.col("column_1").str.starts_with("P_5 ")))

st.write(f"""
I risultati sono poi stati strutturati in modo che il programma trec-eval possa funzionare e fornirci i risultati di rilevanza.
Nel nostro caso usiamo solo map e P_5 che corrispondono a:
- map = {dati[0, 2]}
- P_5 = {dati[1, 2]}
""")


st.markdown("## Risultati delle query")

st.markdown("""
Vediamo le 100 query usate per testing performano in assenza di QE.
A seguire verr\u00e0 riportato un grafico che mostra l'andamento di map e P_5 per le query considerate.
""")

#Creazione grafici relativi al rapporto metrica ~ query
st.markdown("### P@5 ~ Id_Q")
#leggo dati
query = pl.read_csv("./Data/Eval_Queries/bm25_evalQ.txt", has_header= True)
# Imposto slider per fare una selezione delle query da mostrare
selected_p = st.slider("Seleziona il valore di query per p@5", min_value=600, max_value=700, value=(600, 700))
#Filtro le query selezionate
filtered_bm = query.filter((query['id_Q'] >= selected_p[0]) & (query['id_Q'] <= selected_p[1]))

# Creo il grafico per p5~query
chart_p5 = alt.Chart(filtered_bm).mark_point().encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_p)),  # Impostazione dei limiti per l'asse x
    y='p_5'
)

st.altair_chart(chart_p5, use_container_width= True)

st.text(""" 
Il grafico di per s\u00e8 non ci racconta molto da solo, stiamo solo vedendo i risultati per il BM25 senza fare confronti.
""")

#Creazione grafici relativi al rapporto metrica ~ query
st.markdown("### map ~ Id_Q")
# Imposto slider per fare una selezione delle query da mostrare
selected_m = st.slider("Seleziona il valore di query per map", min_value=600, max_value=700, value=(600, 700))
#Filtro le query selezionate
filtered_bm = query.filter((query['id_Q'] >= selected_m[0]) & (query['id_Q'] <= selected_m[1]))
# Creare il grafico per map~query
chart_map = alt.Chart(filtered_bm).mark_point().encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_m)),  # Impostazione dei limiti per l'asse x
    y='map'
)

# Visualizzare il grafico
st.altair_chart(chart_map, use_container_width= True)


st.text(""" 
Stessi risultati di P@5, alcune query performano meglio di altre. 
""")

st.markdown("Abbiamo esaminato i risultati di base; se il nostro metodo alternativo funziona, ci aspettiamo che le query migliorino in termini di precisione. L'obiettivo ideale è che le query con una precisione superiore a una certa soglia rimangano invariate, mentre quelle sotto tale soglia vengano ottimizzate")