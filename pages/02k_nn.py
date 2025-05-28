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
  page_title="bm25kNN"
)

st.markdown("## Come funziona k-nn ")
st.markdown("""
L’algoritmo k-nearest neighbors (k-NN) è una tecnica di apprendimento supervisionato che assegna una classe a un campione in base alle etichette dei suoi k vicini più prossimi nel dataset. Il funzionamento si basa sul calcolo della distanza (tipicamente euclidea) tra il punto da classificare e tutti i punti del dataset, sull’identificazione dei k più vicini, e infine sull’assegnazione della classe in base al voto di maggioranza. Il valore di k è un parametro chiave: valori piccoli possono portare a scelte instabili, mentre valori grandi possono introdurre eccessiva generalizzazione.

Nel presente lavoro non viene utilizzato k-NN per la classificazione, ma ci si affida all’implementazione NearestNeighbors fornita dalla libreria scikit-learn per individuare i k vicini più prossimi a ciascun elemento. Questa implementazione non esegue un vero e proprio clustering, ma si limita a restituire, per ogni punto dato, gli indici e le distanze rispetto ai suoi k vicini nel dataset di riferimento.
            
Questo sistema sar\u00E0 la base per ottenere dai risultati della query iniziale dei descrittori aggiuntivi per fare QE.
""")

st.markdown("### Espansione basata su k-NN")

st.write("""
I documenti che compaiono come vicini in almeno `min_doc_c` casi vengono considerati dominanti.
Se nessuno soddisfa questo criterio, la query non viene espansa.

Per ogni documento dominante si aggregano i suoi vicini in un *giga-documento*, che viene indicizzato separatamente.
La query originale viene quindi rieseguita su questo indice, e si seleziona il *giga-documento* più simile.
Da questo documento si estraggono i termini utili all'espansione della query.
""")

st.markdown("### Selezione dei termini espansivi")

st.write("""
Dal *giga-documento* selezionato si calcolano i vettori di termine. Per ogni termine $t$, si assegna uno score TF-IDF
rispetto all’intera collezione secondo la formula:
""")

st.latex(r"""
\text{score}(t) = tf(t) \cdot \log_{10} \left( \frac{N}{df(t)} \right)
""")

st.write("""
- $tf(t)$: frequenza del termine nel *giga-documento*  
- $df(t)$: numero di documenti della collezione in cui compare $t$  
- $N$: numero totale di documenti nella collezione
""")

st.write("""
I termini vengono ordinati per score e vengono selezionati i primi `top_n`.
In questo metodo ogni termine espansivo riceve un peso fisso $c_0(t_i) = 1$,
mentre l’interpolazione con i termini originali è controllata solo dal parametro $\\lambda$.
""")

st.markdown("# Implementazone")
st.markdown("Di seguito viee riportato il frammento di codice repsonsabile per la QE e l'uso del k-nn. L'implementazione dell'algoritmo di clustering non viene affrontato in qeusto progetto, viene usata l'implementazione presente nel pacchetto __scikit-learn__")
codice = """
# Calcola la DTM, fa clustering e crea la query espansa
# @param esClient connessione al server ElasticSearch   IP
# @param indexName nominativo indice		            IP
# @param querytext query originale.                     IP
# @param params parametro composto da:                  IP
#         -  first_ret numero documenti pseudo-rilevanti 
#         -  n_neighbours numero di vicini, fissato a 5                                                        
#         -  min_doc_c soglia per documenti dominanti                     
#         -  top_n numero di descrittori espansivi                               
#         -  lambda peso da dare ai nuovi descrittori                           
# @return documenti reperiti dalla query espansa        OR
def search_QE_knn(esClient, indexName, querytext, params):

  querytext = preprocess(querytext)
  query_original = {
  	"size": params.get('first_ret', 10),
  	"query": {
  		"match": {
  			"_content": querytext
  		}
  	}
  }
  response = esClient.search(index=indexName, 
                             body=query_original)
  def get_documents(response):
    return [hit['_source']['_content'] 
           for hit in response['hits']['hits']]
  
  documents = get_documents(response) 
  vectorizer = TfidfVectorizer(preprocessor= preprocess,
                               tokenizer=str.split,
                               lowercase=False
  ) #matrice di ocnteggio token
  dtm = vectorizer.fit_transform(documents)  #crea la DTM
  dtm_normalized=normalize(dtm, norm='l2')#Normalizza DTM
  
  #cerca i vicini 
  neigh = NearestNeighbors(
    n_neighbors=params.get('n_neighbours', 5),
    metric="cosine")
  neigh.fit(dtm_normalized)
  
  #calcolo dei cluster stimati
  test=dtm_normalized
  fitted_clusters = neigh.kneighbors(test,
    n_neighbors=params.get('n_neighbours', 5),
    return_distance=False
  )
  
  #occorrenze di un documento in un cluster
  keys = [i for i in range(0, dtm_normalized.shape[0])]
  dominant_docs = dict.fromkeys(keys,0)
  for cluster in fitted_clusters:
    for doc in cluster:
        dominant_docs[doc] += 1
  
  #id dei documenti rilevanti
  def get_dominant_id(doc_freq, criterion):
    return [key for key, 
        value in doc_freq.items() if value > criterion]
  
  dominant_ids=get_dominant_id(dominant_docs,
                               params.get('min_doc_c',6)
                               )
  if not dominant_ids: #reperimento senza QE
     print("! Nessun documento dominante !")
     query_original = {
       "size": 10,
       "query": {
         "match": {
           "_content": querytext
         }
       }
     }
     return esClient.search(index=indexName, 
                            body=query_original)
  
  #fusione dei documenti in accordo ai cluster
  giga_docs = []
  for cluster in fitted_clusters[dominant_ids]:
  cluster_docs = np.array(documents)[cluster]
  giga_docs.append(" ".join(cluster_docs))
  
  #Indicizazzione dei cluster
  indexName_supporto = "small_index"
  esClient.options(ignore_status=[400])
    .indices.create(index=indexName_supporto)
  
  def index_document(this_id, record):
    record = {"id": this_id, "text": record}
    esClient.index(index=indexName_supporto,
                   id=this_id, document=record)
  
  with ThreadPoolExecutor() as executor:
    for this_id, record in enumerate(giga_docs):
      executor.submit(index_document, this_id, record)
  
  query_cluster = {'match': {"text": querytext }}
 
  def get_first_id(query_dict):
    return esClient.search(index=indexName_supporto,
                           query=query_dict,
                           size=1)["hits"]["hits"]
  
  first_id = get_first_id(query_cluster)
  
  i = 0
  time_sleep = 0
  while not first_id: #fino a quando ottiene un risultato
    i += 1
    time_sleep = i/10 
    if i > 120:
      raise ValueError(f"Query effettuata {i} volte")
    sleep(time_sleep) # per non sovraccaricare il server
    first_id = get_first_id(query_cluster)
  first_id = first_id[0]["_id"]
  
  tv = esClient.termvectors(index = indexName_supporto,  
                            id = first_id,
                            fields = "text",
                            term_statistics=True)
  
  #numero di documenti indicizzati
  N_collezione = esClient.count(index=indexName)['count'] 
  
  terms = dict(tv)["term_vectors"]["text"]["terms"]
  
  #calcola i tf-idf per ogni termine
  tf_idfs = {key: 
    value["term_freq"] * log10(
      N_collezione / value["doc_freq"]
    ) for key, value in terms.items()}
  
  #seleziona i $top_n valori piu grandi
  top_n = nlargest(params.get('top_n', 25),
                   tf_idfs.items(), 
                   key=lambda item: item[1])
  
  # Definizione pesi dei nuovi e vecchi termini
  weighted_terms = [
   { "term": {
       "_content": {
         "value": key,
           "boost": 
             (1-params.get('lambda',.7),2)
       }
     }
   } for key, value in top_n]
  
  weighted_terms.extend([
   { "term": {
       "_content": {
         "value": key,
           "boost":
             1 * params.get('lambda',.7)
       }
     }
    } for key in querytext.split(" ")
                        ])
  
  # query espansa 
  query_expanded = {
    "size": 10,
  	"query": {
      "function_score": {
  	    "query": {
  		  "bool": {
  		    "should": weighted_terms
  		  }
  		}
  	  }
  	}
  }
  
  qe_response = esClient.search(index=indexName,
                                body=query_expanded)
  return qe_response	
"""
st.code(codice, language='python')

st.markdown("# Risultati e parametri")
st.markdown("""Ci sono stati diversi parametri da ottimizzare:
  - **first_ret**: numero di documenti dominanti;
  - **min_doc_c**: numero minimo occorrenze per documento dominante;
  - **top_n**: numero di descrittori da estrarre e aggiungere alla query iniziale;
  - **$\\lambda$**: peso da dare ai nuovi descrittori nella query espansa.

Sono riportate le configurazioni che massimizzano il miglioramento di ogni metrica:
1. map:
  - **first_ret** = 10;
  - **min_doc_c** = 4;
  - **top_n** = 10;
  - **$\\lambda$** = 0.9.
            
2. P@5:
  - **first_ret** = 50;
  - **min_doc_c** = 5;
  - **top_n** = 10;
  - **$\\lambda$** = 0.9.	
            
3. nDCG:
  - **first_ret** = 10;
  - **min_doc_c** = 5;
  - **top_n** = 10;
  - **$\\lambda$** = 0.9.	
""")

st.markdown(""" ## Giustificazione scelta parametri""")
st.markdown("Mostriamo come variano le metriche scelte per l'analisi, cerchiamo in particolare se esiste una configurazione maggiormente predisposta per migliorare **map**, **P@5** o **nDCG**")

#Carico i dati relativi ai risultati complessivi
bm25 = pl.read_csv("./Data/EVAL_TEST_Q/base/Eval_Q_QE_base_combined.csv", has_header= True, separator=",")

knn = pl.read_csv("./Data/EVAL_TEST_Q/knn/Eval_Q_QE_knn_combined.csv", has_header= True, separator=",")



# Calcolo delle medie
mean_base = bm25[["map", "p_5", "ndcg_10"]].mean()
mean_knn = knn[["map", "p_5", "ndcg_10"]].mean()

# Calcolo delle medie con Polars  float scalari
map_bm25 = mean_base.select("map").item()
p5_bm25 = mean_base.select("p_5").item()
ndcg_bm25 = mean_base.select("ndcg_10").item()

map_knn = mean_knn.select("map").item()
p5_knn = mean_knn.select("p_5").item()
ndcg_knn = mean_knn.select("ndcg_10").item()

# Dati per Plotly
methods = ["BM25", "k-NN"]
maps = [map_bm25, map_knn]
p5s = [p5_bm25, p5_knn]
ndcgs = [ndcg_bm25, ndcg_knn]

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
    title="Confronto tra BM25 e k-NN",
    margin=dict(l=0, r=0, b=0, t=30)
)

# Visualizzazione in Streamlit
st.plotly_chart(fig, use_container_width=True)



st.markdown("""Il grafico a dispersione mostra dei risultati interessanti. In particolare, il miglior risultato si trova nell'angolo in alto a destra, verso di noi, indicando una configurazione ottimale.
            
Il BM25 si posiziona nella parte superiore destra del grafico, lontano dalla configurazione del k-NN. Questo suggerisce che, nel nostro contesto, il BM25  rappresenta la soluzione migliore rispetto al k-NN.
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

filtered_knn = knn.filter(
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
chart_knn = alt.Chart(filtered_knn).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='p_5',
		color=alt.Color('method', 
                    scale=alt.Scale(range=['#83C9FF', '#FFABAB']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)

st.altair_chart(chart_bm+chart_knn, use_container_width= True)


# Seleziona solo le colonne di interesse
q_bm = bm25.select(["id_Q", "p_5"])  
q_knn = knn.select(["id_Q", "p_5"])    

# Merge tra BM25 e KNN su id_Q
q_merged = q_bm.join(q_knn, on="id_Q", suffix="_knn")

# Calcolo della differenza KNN - BM25
q_merged = q_merged.with_columns(
    (pl.col("p_5_knn") - pl.col("p_5")).alias("difference")
)

# Estrazione massimo e minimo miglioramento
max_diff = q_merged["difference"].max()
min_diff = q_merged["difference"].min()

# Trovo i punti di massimo e minimo
max_index = q_merged["difference"].arg_max()
min_index = q_merged["difference"].arg_min()


st.markdown(f"""### Prendiamo per esempio le seguenti query:
- {max_index + OFFSET}: P@5 senzas QE di {bm25[max_index, "p_5"]} mentre usando QE otteniamo {knn[max_index,"p_5"]}. Il nostro meteodo ha funzionato egregiamente, anche se questo \u00E8 un caso di esempio dove viene massimizzata la differenza.
- {min_index + OFFSET}: P@5 senza QE di {bm25[min_index, "p_5"]}, mentre usando QE P@5 diventa di {knn[min_index, "p_5"]}. Ovvero vengono reperiti zero documenti rilevanti
""")



st.markdown("## MAP ~ Id_Q" "")
#Creo slider per selezionare le query che mi interessano
selected_x = st.slider("Seleziona il valore di query per MAP", min_value=MIN_SLIDE, max_value=MAX_SLIDE, value=(MIN_SLIDE, MAX_SLIDE))

#Filtro le query in bse allo slider
filtered_bm = bm25.filter(
    (pl.col("id_Q") >= selected_x[0]) & (pl.col("id_Q") <= selected_x[1])
).select(["id_Q", "map", "method"])

filtered_knn = knn.filter(
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
chart_knn = alt.Chart(filtered_knn).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='map',
		color=alt.Color('method', 
                    scale=alt.Scale(range=['#83C9FF', '#FFABAB']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)

st.altair_chart(chart_bm+chart_knn, use_container_width= True)


# Seleziona solo le colonne di interesse
q_bm = bm25.select(["id_Q", "map"])  
q_knn = knn.select(["id_Q", "map"])    

# Merge tra BM25 e KNN su id_Q
q_merged = q_bm.join(q_knn, on="id_Q", suffix="_knn")

# Calcolo della differenza KNN - BM25
q_merged = q_merged.with_columns(
    (pl.col("map_knn") - pl.col("map")).alias("difference")
)

# Estrazione massimo e minimo miglioramento
max_diff = q_merged["difference"].max()
min_diff = q_merged["difference"].min()

# Trovo i punti di massimo e minimo
max_index = q_merged["difference"].arg_max()
min_index = q_merged["difference"].arg_min()


st.markdown(f"""### Prendiamo per esempio le seguenti query:
- {max_index + OFFSET}: MAP senza QE di {bm25[max_index, "map"]} mentre usando QE otteniamo {knn[max_index,"map"]}. Il nostro meteodo ha funzionato.
- {min_index + OFFSET}: MAP senza QE di {bm25[min_index, "map"]}, mentre usando QE MAP diventa di {knn[min_index, "map"]}. Ovvero vengono reperiti zero documenti rilevanti ancora una volta.
""")

st.markdown("## nDCG ~ Id_Q" "")
#Creo slider per selezionare le query che mi interessano
selected_x = st.slider("Seleziona il valore di query per nDCG", min_value=MIN_SLIDE, max_value=MAX_SLIDE, value=(MIN_SLIDE, MAX_SLIDE))

#Filtro le query in bse allo slider
filtered_bm = bm25.filter(
    (pl.col("id_Q") >= selected_x[0]) & (pl.col("id_Q") <= selected_x[1])
).select(["id_Q", "ndcg_10", "method"])

filtered_knn = knn.filter(
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
chart_knn = alt.Chart(filtered_knn).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='ndcg_10',
		color=alt.Color('method', 
                    scale=alt.Scale(range=['#83C9FF', '#FFABAB']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)

st.altair_chart(chart_bm+chart_knn, use_container_width= True)


# Seleziona solo le colonne di interesse
q_bm = bm25.select(["id_Q", "ndcg_10"])  
q_knn = knn.select(["id_Q", "ndcg_10"])    

# Merge tra BM25 e KNN su id_Q
q_merged = q_bm.join(q_knn, on="id_Q", suffix="_knn")

# Calcolo della differenza KNN - BM25
q_merged = q_merged.with_columns(
    (pl.col("ndcg_10_knn") - pl.col("ndcg_10")).alias("difference")
)

# Estrazione massimo e minimo miglioramento
max_diff = q_merged["difference"].max()
min_diff = q_merged["difference"].min()

# Trovo i punti di massimo e minimo
max_index = q_merged["difference"].arg_max()
min_index = q_merged["difference"].arg_min()


st.markdown(f"""### Prendiamo per esempio le seguenti query:
- {max_index + OFFSET}: nDCG senza QE di {bm25[max_index, "ndcg_10"]} mentre usando QE otteniamo {knn[max_index,"ndcg_10"]}. Il metodo ha funzionato.
- {min_index + OFFSET}: nDCG senza QE di {bm25[min_index, "ndcg_10"]}, mentre usando QE nDCG diventa di {knn[min_index, "ndcg_10"]}. Ovvero vengono reperiti zero documenti rilevanti ancora una volta.
""")


st.text("Notiamo quindi nonostante alcuni miglioramenti considerevoli relativi alle singole query, ce ne sono altrettanti peggiori. E mediamente il metodo peggiora l'efficicacia del reperimento")