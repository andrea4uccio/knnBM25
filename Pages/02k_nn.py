import altair as alt
import polars as pl
import streamlit as st


# Imposta il titolo pagina che viene fuori nel browser
st.set_page_config(
    page_title="bm25kNN"
)




st.markdown("## Come funziona k-nn ")
st.markdown("""
Speighiamo in breve il concetto dietro il funzionamento di  k-nn.
				 
L'algoritmo di classificazione k-NN (k-nearest neighbors) \u00E8 un metodo di apprendimento supervisionato che assegna una classe a un campione in base alle classi dei suoi k vicini pi\u00F9 prossimi nel dataset di addestramento. Funziona in questo modo:

1. Calcolo della distanza: Per ogni nuovo dato che deve essere classificato, l'algoritmo calcola la distanza tra il punto da classificare e tutti i punti nel dataset di addestramento (di solito usando la distanza euclidea).

2. Identificazione dei vicini: Si selezionano i k punti pi\u00F9 vicini al punto da classificare, dove k \u00E8 un parametro scelto dall'utente.

3. Voto di maggioranza: La classe assegnata al punto da classificare \u00E8 la classe pi\u00F9 frequente tra i k vicini selezionati.

Scelta di k: Il valore di k influisce sul risultato: valori troppo piccoli possono essere sensibili al rumore, mentre valori troppo grandi possono generalizzare troppo e perdere dettagli importanti.

QUesto sistema sar\u00E0 la base per ottenere dai risultati della query iniziale dei descrittori aggiuntivi per fare QE.
""")

st.markdown(" ## k-nn e QE")
st.markdown("""
L'approccio che stiamo usando \u00E8 simile alla pseudo-relevance feedback. Nella pseudo-relevance feedback si assume che i documenti in posizione piu alta come rilevanti.
L'idea quindi cambia, usiamo un cluster di documenti per trovare documenti dominanti per il set reperito iniziale e ripetutamente fornire i documenti per enfatizzare gli argomenti principali di una query. 
""")

st.markdown("## Problema")
st.markdown("""
I documenti reperiti in posizione alta contengono rumore, se la P@10 \u00E8 0.5, di 10 documenti, 5 sono non rilevanti.
Il che puo provocare un drift dei risultati delle query che invece di avvicinarsi a documenti rilevanti si puo avvicinare a quelli non rilevanti.
Trovare il cluster ottimale \u00e8 difficile, quindi useremo una serie di gruppi rilevanti per il feedback permettendo cluster sovrapponibili per i top-retrieved e alimentando ripeturamente i documenti dominantiche appaiono in piu cluster di alto rango ci aspettiamo che QE possa portare a risultati piu precisi.
la motivazione dell'utilizzo di cluster deriva dal fatto che i documenti inposizione pi\u00F9 alta hanno un ordinamento "query-oriented" che nono considera la relazione tra documenti.
Il problema post in precedenza \u00E8 come scegliere i termini di espansione, che devono essere vicini alla quey. Selezionando ripetutamente i documenti dominanti risolviamo questa difficolta.
""")

st.markdown("## Funzionamento")
st.markdown("""
**Documento dominante** : Assumiamo un documento dominanto per una query quel documento con un ottima rappresentazione dei topics di una qury, ovvero uno con diversi vicini con alta similarit\u00E0. Nel nostro caso di cluster sovrapposti un documento dominante sar\\U00E0 quello che appare in molteplici cluster con rango elevato. Dal documento dominante rilevato possiamo ricavare i temmini di espansioneche reperirira documenti rletivi a tutti i sottitopics.
1. I documenti vengon o riperiti data una query usando l'API search() di elasticsearch che di base usa il BM25 senza QE
2. Vengono generati dei cluster di k-nearest vicini per i primi N docuimenti reperiti pr trovare documenti dominanti. 
3. In k-NN ogni docunmento hja un ruolo centrale nel creare il proprio cluster con k vicini per similarit\u00e0.
4. Il rango viene deciso da un language model basato su cluster.
5. I termini di espansione vengono selezionati usando il modello di rilevanza per ogni documento nei top-ranked cluster.			 
""")

st.markdown("# Implementazone")
st.markdown("Di seguito viee riportato il frammento di codice repsonsabile per la QE e l'uso del k-nn. L'implementazione dell'algoritmo di clustering non viene affrontato in qeusto progetto, viene usata l'implementazione presente nel pacchetto __scikit-learn__")
codice = """
# Calcola la DTM, fa clustering e crea la query espansa
# @param server connessione al server ElasticSearch                                   IP
# @param INDEXNAME nominativo indice sul quale si vuole eseguire la ricerca           IP
# @param N_collezione numero di documenti indicizzati in $INDEXNAME                   IP
# @param query_text query originale.                                                  IP
# @param first_query_size numero di documenti che si vogliono reperire con $querytext IP
# @param k numero di vicini                                                           IP
# @param min_occurrences_cluster soglia per determinare documenti dominanti           IP
# @param e numero di descrittori nella query espansa                                  IP
# @param Lambda peos da dare ai nuovi descrittori                                     IP
# @return documenti reperiti dalla query espansa                                      OR
def QE_kNN(server, INDEXNAME, N_collezione, querytext, first_query_size, k, min_occurrences_cluster, e, Lambda):

  query_original = { 'match': { "_content": querytext } }
  response = server.search(index=INDEXNAME, query=query_original, size=first_query_size)

  def get_documents(response):
    return [hit['_source']['_content'] for hit in response['hits']['hits']]

  documents = get_documents(response)
  vectorizer = TfidfVectorizer(stop_words='english')
  dtm = vectorizer.fit_transform(documents)
  # Normalize the DTM
  dtm_normalized = normalize(dtm, norm='l2')

  neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
  # Fit the classifier with the entire dataset (note that X should be the whole DataFrame)
  neigh.fit(dtm_normalized)

  # calcolo dei cluster stimati
  test=dtm_normalized
  # ???? threshold = 0.25 ???
  fitted_clusters = neigh.kneighbors(test, n_neighbors=k, return_distance=False)

  # calcolo del numero di volte in cui un documento appare in un cluster
  keys = [i for i in range(0,dtm_normalized.shape[0])]
  dominant_docs = dict.fromkeys(keys,0)
  for cluster in fitted_clusters:
    for doc in cluster:
      dominant_docs[doc] += 1

  # id dei documenti rilevanti
  def get_dominant_id(doc_freq, criterion):
    return [key for key, value in doc_freq.items() if value > criterion]

  dominant_ids = get_dominant_id(dominant_docs, min_occurrences_cluster)
  if not dominant_ids:
    print("! Nessun documento dominante !")
    return server.search(index=INDEXNAME, query=query_original, size=10)

  # fusione dei documenti in accordo ai cluster
  giga_docs = []
  for cluster in fitted_clusters[dominant_ids]:
    cluster_docs = np.array(documents)[cluster]
    giga_docs.append(" ".join(cluster_docs))

  # Indicizazzione dei cluster
  Indexname_supporto = "small_index"
  server.indices.create(index=Indexname_supporto, ignore=400)
  def index_document(this_id, record):
    record = {"id": this_id, "text": record}
    server.index(index=Indexname_supporto, id=this_id, document=record)

  with ThreadPoolExecutor() as executor:
    for this_id, record in enumerate(giga_docs):
      executor.submit(index_document, this_id, record)

  query_cluster = { 'match': { "text": querytext } }
  
  def get_first_id(query_dict):
    sleep(2)
    return server.search(index=Indexname_supporto, query=query_dict, size=1)["hits"]["hits"]

  #print("Tentativo numero: 1")
  first_id = get_first_id(query_cluster)
  #print("len:",len(first_id))

  i = 1
  while not first_id:
    i += 1
    sleep(i)
    #print("Tentativo numero:",i)
    first_id = get_first_id(query_cluster)
    if i > 10:
      print(server.search(index=Indexname_supporto, query=query_cluster, size=1)["hits"]["hits"])
      raise ValueError(f"Query effettuata {i} volte")

  #print("Numero di richieste mandate al server:",i)
  first_id = first_id[0]["_id"]

  tv = server.termvectors(index  = Indexname_supporto,         # term vector dall'indice
                          id     =  first_id,        # per il documento documento
                          fields = "text",            # con questi campi
                          term_statistics=True)
  #pprint(dict(tv)["term_vectors"]["text"]["terms"])

  terms = dict(tv)["term_vectors"]["text"]["terms"]
  tf_idfs = {key: value["term_freq"] * log10(N_collezione / value["doc_freq"]) for key, value in terms.items()}
  #pprint(tf_idfs)

  top_e = nlargest(e, tf_idfs.items(), key=lambda item: item[1])

  # Define your weighted terms
  weighted_terms = [{"term": {"_content": {"value": key, "boost": 1*(1-Lambda)}}} for key, value in top_e]
  weighted_terms.extend([{"term": {"_content": {"value": key, "boost": 1*Lambda}}} for key in querytext.split(" ")])
  #pprint(weighted_terms)

  # Create the query using function_score
  query_expanded = {
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

  # Execute the search query
  qe_response = server.search(index=INDEXNAME, body=query_expanded)
  #pprint(response)
  return qe_response

"""
st.code(codice, language='python')

st.markdown("# Risultati e parametri")
st.markdown("""Ci sono stati diversi parametri da ottimizzare:
  - **N**: numero di documenti dominanti;
  - **e**: numero di descrittori da estrarre e aggiungere alla query iniziale;
  - **$\\lambda$**: peso da dare ai nuovi descrittori nella query espansa.

Sono riportate le due configurazioni che si sono distinte maggirmente:
1. Prima configurazione:
  - **N** = 10;
  - **e** = 25;
  - **$\\lambda$** = 0.7.
2. Seconda configurazione:
  - **N** = 10;
  - **e** = 100;
  - **$\\lambda$** = 0.9.	
""")

st.markdown(""" ## Giustificazione scelta parametri""")
st.markdown("Mostriamo come variano le metriche scelte per l'analisi, ovvero **map** e **P@5**")


bm25 = pl.read_csv("./Data/Eval_Test/test_results_bm25_eval.txt", has_header= False, separator="\t")
kn2507 = pl.read_csv("./Data/Eval_Test/test_results_10_25_07_eval.txt", has_header= False, separator="\t")
kn10009 = pl.read_csv("./Data/Eval_Test/test_results_10_100_09_eval.txt", has_header= False, separator="\t")

datibm = bm25.filter((pl.col("column_1").str.starts_with("map")) | (pl.col("column_1").str.starts_with("P_5 ")))
dati07 = kn2507.filter((pl.col("column_1").str.starts_with("map")) | (pl.col("column_1").str.starts_with("P_5 ")))
dati09 = kn10009.filter((pl.col("column_1").str.starts_with("map")) | (pl.col("column_1").str.starts_with("P_5 ")))

datibm[0,0] = "map_BM25"
datibm[1,0] = "P@5_BM25"
dati07[0,0] = "map_first"
dati07[1,0] = "P@5_first"
dati09[0,0] = "map_second"
dati09[1,0] = "P@5_second"

df_unito = pl.concat([datibm, dati07, dati09])
df_unito = df_unito.rename({"column_1" : "configurazione"})
# Creare il grafico per p5~query
chart = alt.Chart(df_unito).mark_bar().encode(
  	x=alt.X('configurazione', title = "Configurazione considerata"),  # Impostazione dei limiti per l'asse x
    y=alt.Y('column_3', title = "valore precisione"),
		color = 'configurazione'		
).properties(
	title = "Grafico a barre precisione ~ configurazione usata (The higher the better)"
)

# Visualizzare il grafico
st.altair_chart(chart, use_container_width= True)

st.markdown("""
Dai grafici notiamo che i risultati relativi a tutte le query nell'insieme di test migliorano rispetto la configurazione senza QE.
Possiamo inoltre osservare che la **configurazione 1** migliora il map mentre **configurazione 2** migliora la P@5.
Prima di dire che il metodo funzioni \u00e8 opportuno fare un analisi per ogni query. 
""")



st.markdown("# Modello che massimizza P@5")
st.markdown("""Prendiamo in analisi il modello che massimizza la metrica P@5, ovvero quello con parametri $lambda = 0.9$, $e = 100$ ed $N$ = 10. 
Valutiamo come le singole query vengono influenzate dall'uso di questo modello e quali migliorano o peggiorano""")


querybm25 = pl.read_csv("./Data/Eval_Queries/bm25_evalQ.txt", has_header= True)
query09 = pl.read_csv("./Data/Eval_Queries/10_100_09_evalQ.txt", has_header= True)


selected_x = st.slider("Seleziona il valore di query per p@5", min_value=600, max_value=700, value=(600, 700))
filtered_bm = querybm25.filter((querybm25['id_Q'] >= selected_x[0]) & (querybm25['id_Q'] <= selected_x[1]))
filtered_09 = query09.filter((query09['id_Q'] >= selected_x[0]) & (query09['id_Q'] <= selected_x[1]))

chart_bm = alt.Chart(filtered_bm).mark_point(color = '#FF6347', filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='p_5', 
		color = "method"
)

chart_09 = alt.Chart(filtered_09).mark_point(color = '#32CD32', filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='p_5',
		color = "method"
)

st.altair_chart(chart_bm+chart_09, use_container_width= True)

st.text("""
Il grafico non \u00e8 molto suggestivo, per\u00f2 possiamo notare che alcune query rimangono invariate, altre invece cambiano sensibilmente.
""" )

st.markdown(f"""### Prendiamo per esempio le seguenti query:
- 622: "Price Fixing" con un P@5 senzas QE di {querybm25[21,1]} mentre usando QE otteniamo {query09[21,1]}. Il nostro meteodo ha funzionato egregiamente, anche se questo \u00E8 un caso di esempio dove viene massimizzata la differenza.
- 626: "Human Stampede" con P@5 senza QE di {querybm25[25,1]}, mentre usando QE P@5 diventa di {query09[25,1]}. Ovvero viene dimezzata la precisione. 
""")


st.markdown("# Modello che massimizza map")
st.markdown("""Prendiamo in analisi il modello che massimizza la metrica map, ovvero quello con parametri $lambda = 0.7$, $e = 25$ ed $N$ = 10. 
Valutiamo come le singole query vengono influenzate dall'uso di questo modello e quali migliorano o peggiorano""")


query07 = pl.read_csv("./Data/Eval_Queries/10_25_07_evalQ.txt", has_header= True)

selected_x = st.slider("Seleziona il valore di query per map", min_value=600, max_value=700, value=(600, 700))
filtered_bm = querybm25.filter((querybm25['id_Q'] >= selected_x[0]) & (querybm25['id_Q'] <= selected_x[1]))
filtered_07 = query07.filter((query07['id_Q'] >= selected_x[0]) & (query07['id_Q'] <= selected_x[1]))

chart_bm = alt.Chart(filtered_bm).mark_point(color = '#FF6347', filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='map', 
		color = "method"
)

chart_07 = alt.Chart(filtered_07).mark_point(color = '#32CD32', filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='map',
		color = "method"
)

st.altair_chart(chart_bm+chart_07, use_container_width= True)

st.text("""
Il grafico non \u00e8 molto suggestivo, per\u00F2 possiamo notare che alcune query rimangono invariate, altre invece cambiano sensibilmente.
""" )

st.markdown(f"""### Prendiamo per esempio le seguenti query:
- 607: "Human genetic code" con un map senzas QE di {querybm25[6,2]} mentre usando QE otteniamo {query09[6,2]}. Il nostro meteodo ha funzionato egregiamente, anche se questo \u00E8 un caso di esempio dove viene massimizzata la differenza.
- 626: "Human Stampede" con map senza QE di {querybm25[25,2]}, mentre usando QE mapdiventa di {query09[25,2]}. meno della meta della precisione. 
""")


st.markdown("""Si possono fare ulteriori considerazioni sul perch\u00e9 alcune query hanno funzionato meglio di altre ma non \u00E8 l'obbiettivo di questo progetto.""")