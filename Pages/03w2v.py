import altair as alt
import polars as pl
import streamlit as st
import plotly.graph_objects as go

# Imposto costanti per non avere numeri 
OFFSET = 601 # imposta offset tra indice e id_Q, sara` immediato quando comparira`
MIN_SLIDE = 600 # valore minimo per slider
MAX_SLIDE = 700 # valroe massimo per slider


# Imposta il titolo pagina che viene fuori nel browser
st.set_page_config(
  page_title="bm25w2v"
)

st.markdown("## Come funziona k-nn ")
st.markdown("""
L’algoritmo Word2Vec è un modello non supervisionato di tipo predittivo che apprende rappresentazioni vettoriali (embedding) delle parole a partire da un corpus testuale. L’idea di base è che parole che appaiono in contesti simili avranno rappresentazioni simili nello spazio vettoriale.

Il modello può essere addestrato con due architetture principali:

1. CBOW (Continuous Bag of Words): predice una parola centrale a partire dal contesto (le parole circostanti).

2. Skip-gram: predice il contesto dato una parola centrale.

Per questo lavoro \`e stato scelto il modello skip-gram che sie rivelato miglior per l'accuratezza semantica.
            
Durante l’addestramento, Word2Vec ottimizza i vettori delle parole in modo che quelle che co-occorrono frequentemente siano vicine tra loro nello spazio semantico. Questo consente di catturare relazioni di similarità semantica tra le parole.

Nel contesto dell'espansione di query, i vettori Word2Vec vengono utilizzati per trovare parole semanticamente vicine ai termini della query originale. Questi termini simili vengono considerati descrittori aggiuntivi che possono migliorare il recupero, ampliando il significato della query originale con termini affini appresi dal modello.

Questo approccio consente di superare i limiti lessicali del matching esatto, favorendo una corrispondenza semantica tra i documenti e le query.
""")

st.markdown("## Word2Vec e QE")
st.markdown("""
L'approccio che stiamo usando \u00E8 la pseudo-relevance feedback. Nella pseudo-relevance feedback si assume che i documenti in posizione pi\u00F9 alta come rilevanti.
Usiamo i primi documenti ritenuti pseudo-rilevanti, parametro da ottimizzare, per addestrare localmente il modello Word2Vec. 
""")

st.markdown("## Funzionamento")
st.markdown("""
Nel metodo di espansione basato su Word2Vec, il modello viene addestrato localmente per ogni query, utilizzando i documenti recuperati inizialmente tramite BM25. Questa scelta consente di adattare dinamicamente lo spazio semantico alle specificità di ciascuna query, evitando l’uso di un modello globale addestrato su un corpus eterogeneo.

Una volta indicizzati i documenti e ottenuti i risultati iniziali, il modello Word2Vec viene addestrato sul solo contenuto testuale dei documenti ritenuti pseudo-rilevanti. In questo modo, gli embedding prodotti riflettono la distribuzione locale delle parole nel contesto della query. Per ogni termine della query originale vengono poi individuati i termini più simili nel vettore semantico, secondo la similarità coseno.

I termini selezionati vengono successivamente filtrati per rimuovere eventuali duplicati o parole già presenti nella query, e utilizzati come termini di espansione. L’obiettivo è quello di ampliare la query iniziale con parole semanticamente affini, migliorando il recupero di documenti pertinenti che utilizzano varianti lessicali o sinonimie non presenti nella formulazione originale.

L’intero processo si basa sulla capacità del modello Word2Vec di catturare relazioni semantiche tra i termini, sfruttando la co-occorrenza locale nei documenti inizialmente recuperati. Questo approccio si è dimostrato particolarmente efficace nel migliorare la copertura semantica della query e, in molti casi, ha portato a un incremento delle metriche di retrieval, come la MAP, P@5 e la nDCG.
""")

st.markdown("# Implementazone")
st.markdown("Di seguito viee riportato il frammento di codice repsonsabile per la QE e l'uso di word2vec. L'implementazione dell'algoritmo di embedding non viene affrontato in questo progetto, viene usata l'implementazione presente nel pacchetto __gensim.models__")
codice = """
# Implementa la ricerca con il modello Word2Vec
# @param esClient connessione al server ElasticSearch   IP
# @param indexname nominativo indice 					IP
# @param query query originale che si vuole espandere   IP
# @param params parametro composto da:														
# - firt_ret : documenti pseudo-rilevanti da considerare
# - top_n    : numero di termini espansivi
# - lambda   : per bilanciare termini originali e espansi
# - window   : grandezza finestra del contesto    
# - min_count: parametro per parole rare       
# - negative : parametro relativo al negative sampling  
# - epochs   : quante volte esaminare l'intero corpus    
# - sample   : sottocampionamento parole frequenti
# @return documenti reperiti dalla query espansa        OR
def search_QE_w2v(esClient, indexname, query, params):
  sta_tim = datetime.now()
  query = preprocess(query)
  
  # Query per reperimento iniziale
  base_query = {
  	"query": {"match": {"_content": query}},
  	"size": params.get('ret', 10)
  }
  response = esClient.search(index=indexname,
  						   body=base_query)
  len_firstRet = len(response['hits']['hits'])
  
  #ottengo il contenuto dei documenti pseudo-rilevanti
  docs = [hit["_source"]["_content"] 
  	   for hit in response["hits"]["hits"]]
  
  #preprocessing per addestrare il modello 
  tokenized_docs = [preprocess(doc).split() 
  				  for doc in docs]
  
  # training modello Word2Vec 
  startTrainTime = datetime.now()
  local_model = Word2Vec(
  sentences   = tokenized_docs,
  vector_size = params.get("vector_size", 75),
  window      = params.get("window", 3),      
  min_count   = params.get("min_count", 1),    
  sg          = 1, #uso Skip-Gram                                        
  negative    = params.get("negative", 5),     
  epochs      = params.get("epochs", 10),      
  sample      = params.get("sample", 1e-3),     
  workers     = 14)
  elapsed_train = datetime.now() - startTrainTime
  
  #ottengo descrittori con cui espandere la query
  selected_terms=get_QE_w2v_balanced(local_model,
  				 			   query, 
  				 			   params.get("top_n", 7))
  
  #definizione pesi dei nuovi e vecchi termini
  original_terms = query.split()
  boosted_original = [
  {"term": {"_content":{"value":term, 
  				  "boost":params.get("lambda",.7)}}}
  for term in original_terms]
  
  boosted_expanded = [
    {"term": 
    	{"_content": 
    	  {"value": term,
  	   "boost": (1 - params.get("lambda",.7)) * score
     	  }
    	  	}
        }
    for term, score in selected_terms
  ]
  
  # Query espansa
  query_expanded = {
    "query": {
      "bool": {
  	  "should": boosted_original + boosted_expanded
  	}
    },
    "size": 10
  }
  log_result((datetime.now()-sta_tim).total_seconds(), 
  			query, elapsed_train.total_seconds(), 
  			len_first= len_firstRet, params=params,
  			top_e= selected_terms)
  return esClient.search(index=indexname,
  	body=query_expanded)
"""
st.code(codice, language='python')







st.markdown("# Risultati e parametri")
st.markdown("""Abbiamo diversi parametri relativi all'implementazione, un gruppo e responsabile per l'addestramento di Word2Vec e un altro `e relativo al QE. Di seguito sonon elecanti tutti i parametri:
 
- **vector_size**: dimensione dei vettori di embedding associati ai termini; valori maggiori permettono rappresentazioni semantiche più dettagliate;  
- **window**: ampiezza della finestra di contesto, ovvero quante parole vicine al termine target vengono considerate durante l’apprendimento;  
- **min_count**: soglia minima di frequenza sotto la quale un termine viene escluso dal vocabolario;  
- **negative**: definisce il numero di esempi negativi da campionare per ogni esempio positivo durante l’addestramento di Word2Vec. Valori più alti migliorano la discriminazione, ma aumentano il costo computazionale.
- **epochs**: numero di iterazioni complete sul corpus durante l’addestramento del modello.
- **first_ret**: numero di documenti pseudo-rilevanti da recuperare per effettuare l'espansione  
- **top_n**: numero di termini descrittori da aggiungere alla query iniziale;  
- **$\\lambda$**: proporzione di peso da assegnare ai descrittori espansi rispetto ai termini originali nella query;  



Sono riportate le configurazioni che si sono distinte maggiormente:

1. **MAP**  
  - **vector_size** = 75  
  - **window**  = 5
  - **min_count**  = 5
  - **negative**  = 15
  - **epochs**  = 10
  - **first_ret** = 10  
  - **top_n** = 50  
  - **$\\lambda$** = 0.9  

2. **P@5**  
  - **vector_size** = 125  
  - **window**  = 5
  - **min_count**  = 5
  - **negative**  = 10
  - **epochs**  = 10
  - **first_ret** = 5  
  - **top_n** = 100  
  - **$\\lambda$** = 0.9  

3. **nDCG**  
  - **vector_size** = 100  
  - **window**  = 5
  - **min_count**  = 7 
  - **negative**  = 15
  - **epochs**  = 10
  - **first_ret** = 25  
  - **top_n** = 100  
  - **$\\lambda$** = 0.9  

""")

st.markdown(""" ## Giustificazione scelta parametri""")
st.markdown("Mostriamo come variano le metriche scelte per l'analisi, cerchiamo in particolare se esiste una configurazione maggiormente predisposta per migliorare **map**, **P@5** o **nDCG**")

#Carico i dati relativi ai risultati complessivi
bm25 = pl.read_csv("./Data/EVAL_TEST_Q/base/Eval_Q_QE_base_combined.csv", has_header= True, separator=",")

w2v = pl.read_csv("./Data/EVAL_TEST_Q/w2v/Eval_Q_QE_w2v_combined.csv", has_header= True, separator=",")

# Calcolo delle medie
mean_base = bm25[["map", "p_5", "ndcg_10"]].mean()
mean_w2v = w2v[["map", "p_5", "ndcg_10"]].mean()

# Calcolo delle medie con Polars  float scalari
map_bm25 = mean_base.select("map").item()
p5_bm25 = mean_base.select("p_5").item()
ndcg_bm25 = mean_base.select("ndcg_10").item()

map_w2v = mean_w2v.select("map").item()
p5_w2v = mean_w2v.select("p_5").item()
ndcg_w2v = mean_w2v.select("ndcg_10").item()

# Dati per Plotly
methods = ["BM25", "w2v"]
maps = [map_bm25, map_w2v]
p5s = [p5_bm25, p5_w2v]
ndcgs = [ndcg_bm25, ndcg_w2v]

fig = go.Figure()

# Aggiunta dei punti
fig.add_trace(go.Scatter3d(
    x=maps,
    y=ndcgs,
    z=p5s,
    mode='markers+text',
    text=methods,
    textposition="top center",
    marker=dict(
        size=8,
        color=['blue', 'red'],
        opacity=0.8
    )
))

fig.update_layout(
    scene=dict(
        xaxis_title='MAP',
        yaxis_title='nDCG@10',
        zaxis_title='P@5'
    ),
    title="Confronto tra BM25 e w2v",
    margin=dict(l=0, r=0, b=0, t=30)
)

# Visualizzazione in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.markdown("""Il grafico mostra che il metodo che implementa Word2Vec migliora notevolmente i risultati rispetto alla baseline, presentadosi come ottimo candidato.
""")

st.markdown("# Andamento metriche per le query")
st.markdown("""Analizziamo varie configurazioni del modello e valutiamo come le singole query vengono influenzate""")


st.markdown("## P@5 ~ Id_Q" "")
#Creo slider per selezionare le query che mi interessano
selected_x = st.slider("Seleziona il valore di query per p@5", min_value=MIN_SLIDE, max_value=MAX_SLIDE, value=(MIN_SLIDE, MAX_SLIDE))

#Filtro le query in bse allo slider
filtered_bm = bm25.filter(
    (pl.col("id_Q") >= selected_x[0]) & (pl.col("id_Q") <= selected_x[1])
).select(["id_Q", "p_5", "method"])

filtered_w2v = w2v.filter(
    (pl.col("id_Q") >= selected_x[0]) & (pl.col("id_Q") <= selected_x[1])
).select(["id_Q", "p_5", "method"])

#creo grafico che mostra andamento della metrica p@5 usando il metodo senza QE
chart_bm = alt.Chart(filtered_bm).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='p_5', 
		color=alt.Color('method', 
                    scale=alt.Scale(range=['#83C9FF', '#FFABAB']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)
#creo grafico che mostra andamento della metrica p@5 usando il metodo con QE che massimizza P@5
chart_w2v = alt.Chart(filtered_w2v).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='p_5',
		color=alt.Color('method', 
                    scale=alt.Scale(range=['#83C9FF', '#FFABAB']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)

st.altair_chart(chart_bm+chart_w2v, use_container_width= True)


# Seleziona solo le colonne di interesse
q_bm = bm25.select(["id_Q", "p_5"])  
q_w2v = w2v.select(["id_Q", "p_5"])    

# Merge tra BM25 e w2v su id_Q
q_merged = q_bm.join(q_w2v, on="id_Q", suffix="_w2v")

# Calcolo della differenza w2v - BM25
q_merged = q_merged.with_columns(
    (pl.col("p_5_w2v") - pl.col("p_5")).alias("difference")
)

# Estrazione massimo e minimo miglioramento
max_diff = q_merged["difference"].max()
min_diff = q_merged["difference"].min()

# Trovo i punti di massimo e minimo
max_index = q_merged["difference"].arg_max()
min_index = q_merged["difference"].arg_min()


st.markdown(f"""### Prendiamo per esempio le seguenti query:
- {max_index + OFFSET}: P@5 senzas QE di {bm25[max_index, "p_5"]} mentre usando QE otteniamo {w2v[max_index,"p_5"]}. Il metodo quadruplica l'efficacia.
- {min_index + OFFSET}: P@5 senza QE di {bm25[min_index, "p_5"]}, mentre usando QE P@5 diventa di {w2v[min_index, "p_5"]}, dimezzandola ma pur ottenendo risultati rilevanti.
""")



st.markdown("## MAP ~ Id_Q" "")
#Creo slider per selezionare le query che mi interessano
selected_x = st.slider("Seleziona il valore di query per MAP", min_value=MIN_SLIDE, max_value=MAX_SLIDE, value=(MIN_SLIDE, MAX_SLIDE))

#Filtro le query in bse allo slider
filtered_bm = bm25.filter(
    (pl.col("id_Q") >= selected_x[0]) & (pl.col("id_Q") <= selected_x[1])
).select(["id_Q", "map", "method"])

filtered_w2v = w2v.filter(
    (pl.col("id_Q") >= selected_x[0]) & (pl.col("id_Q") <= selected_x[1])
).select(["id_Q", "map", "method"])

#creo grafico che mostra andamento della metrica MAP usando il metodo senza QE
chart_bm = alt.Chart(filtered_bm).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='map', 
		color=alt.Color('method', 
                    scale=alt.Scale(range=['#83C9FF', '#FFABAB']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)
#creo grafico che mostra andamento della metrica MAP usando il metodo con QE che massimizza MAP
chart_w2v = alt.Chart(filtered_w2v).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='map',
		color=alt.Color('method', 
                    scale=alt.Scale(range=['#83C9FF', '#FFABAB']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)

st.altair_chart(chart_bm+chart_w2v, use_container_width= True)


# Seleziona solo le colonne di interesse
q_bm = bm25.select(["id_Q", "map"])  
q_w2v = w2v.select(["id_Q", "map"])    

# Merge tra BM25 e w2v su id_Q
q_merged = q_bm.join(q_w2v, on="id_Q", suffix="_w2v")

# Calcolo della differenza w2v - BM25
q_merged = q_merged.with_columns(
    (pl.col("map_w2v") - pl.col("map")).alias("difference")
)

# Estrazione massimo e minimo miglioramento
max_diff = q_merged["difference"].max()
min_diff = q_merged["difference"].min()

# Trovo i punti di massimo e minimo
max_index = q_merged["difference"].arg_max()
min_index = q_merged["difference"].arg_min()


st.markdown(f"""### Prendiamo per esempio le seguenti query:
- {max_index + OFFSET}: MAP senza QE di {bm25[max_index, "map"]} mentre usando QE otteniamo {w2v[max_index,"map"]}. La MAP viene circa raddopiata.
- {min_index + OFFSET}: MAP senza QE di {bm25[min_index, "map"]}, mentre usando QE MAP diventa di {w2v[min_index, "map"]}. Ancora una volta viene quasi dimezzata ma non annullata. MAP >0 anche nel caso pessimo.
""")

st.markdown("## nDCG ~ Id_Q" "")
#Creo slider per selezionare le query che mi interessano
selected_x = st.slider("Seleziona il valore di query per nDCG", min_value=MIN_SLIDE, max_value=MAX_SLIDE, value=(MIN_SLIDE, MAX_SLIDE))

#Filtro le query in bse allo slider
filtered_bm = bm25.filter(
    (pl.col("id_Q") >= selected_x[0]) & (pl.col("id_Q") <= selected_x[1])
).select(["id_Q", "ndcg_10", "method"])

filtered_w2v = w2v.filter(
    (pl.col("id_Q") >= selected_x[0]) & (pl.col("id_Q") <= selected_x[1])
).select(["id_Q", "ndcg_10", "method"])

#creo grafico che mostra andamento della metrica nDCG usando il metodo senza QE
chart_bm = alt.Chart(filtered_bm).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='ndcg_10', 
		color=alt.Color('method', 
                    scale=alt.Scale(range=['#83C9FF', '#FFABAB']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)
#creo grafico che mostra andamento della metrica nDCG usando il metodo con QE che massimizza nDCG
chart_w2v = alt.Chart(filtered_w2v).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='ndcg_10',
		color=alt.Color('method', 
                    scale=alt.Scale(range=['#83C9FF', '#FFABAB']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)

st.altair_chart(chart_bm+chart_w2v, use_container_width= True)


# Seleziona solo le colonne di interesse
q_bm = bm25.select(["id_Q", "ndcg_10"])  
q_w2v = w2v.select(["id_Q", "ndcg_10"])    

# Merge tra BM25 e w2v su id_Q
q_merged = q_bm.join(q_w2v, on="id_Q", suffix="_w2v")

# Calcolo della differenza w2v - BM25
q_merged = q_merged.with_columns(
    (pl.col("ndcg_10_w2v") - pl.col("ndcg_10")).alias("difference")
)

# Estrazione massimo e minimo miglioramento
max_diff = q_merged["difference"].max()
min_diff = q_merged["difference"].min()

# Trovo i punti di massimo e minimo
max_index = q_merged["difference"].arg_max()
min_index = q_merged["difference"].arg_min()


st.markdown(f"""### Prendiamo per esempio le seguenti query:
- {max_index + OFFSET}: nDCG senza QE di {bm25[max_index, "ndcg_10"]} mentre usando QE otteniamo {w2v[max_index,"ndcg_10"]}. Ancora una volta ha quasi quadruplicato la metrica.
- {min_index + OFFSET}: nDCG senza QE di {bm25[min_index, "ndcg_10"]}, mentre usando QE nDCG diventa di {w2v[min_index, "ndcg_10"]}. Purtroppo non sono stati reperiti documenti rilevanti. 
""")


st.text("Notiamo quindi nonostante alcuni miglioramenti considerevoli relativi alle singole query, nonostante i peggiormenti il metodo riesce, per le combinaizoni di parametri che ottimizzano MAP e P@5, a reperire qualche documento rilevante. Mediamente il metodo migliora l'efficicacia del reperimento")