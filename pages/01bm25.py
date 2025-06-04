import altair as alt
import polars as pl
import streamlit as st

# Imposta il titolo pagina che viene fuori nel browser
st.set_page_config(
    page_title="bm25kNN"
)


st.title("Cosa \u00e8 il bm25 ")
st.markdown(r"""
Il **BM25 (Best Matching 25)**  \u00e8 un algoritmo di ranking ampiamente utilizzato nei sistemi di Information Retrieval, e rappresenta una delle implementazioni pi\u00f2 efficaci del modello probabilistico di rilevanza.  
È adottato di default da Elasticsearch e viene utilizzato per assegnare a ogni documento un punteggio che indica quanto \u00e8 pertinente rispetto a una determinata query.

BM25 si basa su due concetti fondamentali:

- **Term Frequency (TF)**: misura quante volte un termine appare in un documento. BM25 applica una funzione di saturazione: l'importanza del termine aumenta con la frequenza, ma in modo decrescente.
- **Inverse Document Frequency (IDF)**: misura l'importanza del termine su tutta la collezione. I termini rari sono considerati pi\u00f2 informativi.

Il punteggio BM25 per un termine $t$ nel documento $d$ \u00e8 definito come:

$$
BM25_{ij} = SAT_{ij} \cdot IDF_j
$$

Dove:
- $IDF_j$ \u00e8 il peso globale del termine $j$, calcolato sull'intera collezione;
- $SAT_{ij}$ \u00e8 il termine di saturazione che modula l’effetto della frequenza del termine nel documento $i$.

La saturazione \u00e8 controllata da due parametri:
- $k_1$: regola la sensibilit\u00e0 alla frequenza del termine (tipico valore: 1.2);
- $b$: regola la normalizzazione rispetto alla lunghezza del documento (tipico valore: 0.75).

Nel presente progetto, i valori di $k_1$ e $b$ non sono stati modificati e corrispondono a quelli predefiniti di Elasticsearch.

---

In termini teorici, BM25 può essere visto come una semplificazione del **Term Relevance Weight (TRW)** del modello probabilistico di retrieval.  

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
bm25 = pl.read_csv("./Data/EVAL_TEST_Q/base/Eval_Q_QE_base_combined.csv", has_header= True, separator=",")


st.write(f"""
I risultati sono poi stati strutturati in modo che il programma trec-eval possa funzionare e fornirci i risultati di rilevanza.
Nel nostro caso usiamo map, P@5 e nDCG@10 che corrispondono a:
- map =     {round(bm25.select(pl.col("map").mean()).item(),4)}
- P@5 =     {round(bm25.select(pl.col("p_5").mean()).item(),4)}
- nDCG@10 = {round(bm25.select(pl.col("ndcg_10").mean()).item(),4)}
""")


st.markdown("## Risultati delle query")

st.markdown("""
Vediamo le 100 query usate per testing performano in assenza di QE.
A seguire verranno riportati dei grafici che mostrano l'andamento delle metriche considerate in relazione alle query.
""")




#Creazione grafici relativi al rapporto metrica ~ query
st.markdown("### map ~ Id_Q")
# Imposto slider per fare una selezione delle query da mostrare
selected_m = st.slider("Seleziona il valore di query per map", min_value=600, max_value=700, value=(600, 700))

#Filtro le query selezionate
filtered_bm_map = bm25.filter(
    (pl.col("id_Q") >= selected_m[0]) & (pl.col("id_Q") <= selected_m[1])
).select(["id_Q","map"])

# Creare il grafico per map~query
chart_map = alt.Chart(filtered_bm_map).mark_point().encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_m)),  # Impostazione dei limiti per l'asse x
    y='map'
)
# Visualizzare il grafico
st.altair_chart(chart_map, use_container_width= True)



#Creazione grafici relativi al rapporto metrica ~ query
st.markdown("### P@5 ~ Id_Q")

# Imposto slider per fare una selezione delle query da mostrare
selected_p = st.slider("Seleziona il valore di query per p@5", min_value=600, max_value=700, value=(600, 700))

#Filtro le query selezionate
filtered_bm_p5 = bm25.filter(
    (pl.col("id_Q") >= selected_p[0]) & (pl.col("id_Q") <= selected_p[1])
).select(["id_Q","p_5"])

# Creo il grafico per p5~query
chart_p5 = alt.Chart(filtered_bm_p5).mark_point().encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_p)),  # Impostazione dei limiti per l'asse x
    y='p_5'
)

st.altair_chart(chart_p5, use_container_width= True)



#Creazione grafici relativi al rapporto metrica ~ query
st.markdown("### nDCG@10 ~ Id_Q")
# Imposto slider per fare una selezione delle query da mostrare
selected_n = st.slider("Seleziona il valore di query per nDCG", min_value=600, max_value=700, value=(600, 700))

#Filtro le query selezionate
filtered_bm_ndcg = bm25.filter(
    (pl.col("id_Q") >= selected_n[0]) & (pl.col("id_Q") <= selected_n[1])
).select(["id_Q","ndcg_10"])

# Creare il grafico per map~query
chart_ndcg = alt.Chart(filtered_bm_ndcg).mark_point().encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_n)),  # Impostazione dei limiti per l'asse x
    y='ndcg_10'
)

# Visualizzare il grafico
st.altair_chart(chart_ndcg, use_container_width= True)



st.text(""" 
I grafici presenti servono solo a mostrare i risultati della baseline. In tutti i grafici notiamo che ci sono query che performano meglio e altre che performano peggio; se il nostro metodo alternativo funziona, ci aspettiamo che le query migliorino in termini di precisione. L'obiettivo ideale \u00e8 che per ogni query si ottengano tutti e solo documenti rilevanti.
""")