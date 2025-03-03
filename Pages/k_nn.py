import altair as alt
import polars as pl
import streamlit as st


# Imposta il titolo pagina che viene fuori nel browser
st.set_page_config(
    page_title="bm25kNN"
)




st.title("Come funziona k-nn ")
st.write("""
Speighiamo in breve il concetto dietro il funzionamento di  k-nn.
				 
L'algoritmo di classificazione k-NN (k-nearest neighbors) è un metodo di apprendimento supervisionato che assegna una classe a un campione in base alle classi dei suoi k vicini più prossimi nel dataset di addestramento. Funziona in questo modo:

1. Calcolo della distanza: Per ogni nuovo dato che deve essere classificato, l'algoritmo calcola la distanza tra il punto da classificare e tutti i punti nel dataset di addestramento (di solito usando la distanza euclidea).

2. Identificazione dei vicini: Si selezionano i k punti più vicini al punto da classificare, dove k è un parametro scelto dall'utente.

3. Voto di maggioranza: La classe assegnata al punto da classificare è la classe più frequente tra i k vicini selezionati.

Scelta di k: Il valore di k influisce sul risultato: valori troppo piccoli possono essere sensibili al rumore, mentre valori troppo grandi possono generalizzare troppo e perdere dettagli importanti.

QUesto sistema sara` la base per ottenere dai risultati della query iniziale dei descrittori aggiuntivi per fare QE.
""")

st.title("k-nn e QE")
st.write("""
L'approccio che stiamo usando e` simile alla pseudo-relevance feedback. Nella pseudo-relevance feedback si assume che i documenti in posizione piu alta come rilevanti.
L'idea quindi cambia, usiamo un cluster di documenti per trovare documenti dominanti per il set reperito iniziale e ripetutamente fornire i documenti per enfatizzare gli argomenti principali di una query. 
""")

st.title("Problema")
st.write("""
				 I documenti reperiti in posizione alta contengono rumore, se la P@10 e` 0.5, di 10 documenti, 5 sono non rilevanti.
				 Il che puo provocare un drift dei risultati delle query che invece di avvicinarsi a documenti rilevanti si puo avvicinare a quelli non rilevanti.
				 Trovare il cluster ottimale e difficile, quindi useremo una serie di gruppi rilevanti per il feedback permettendo cluster sovrapponibili per i top-retrieved e alimentando ripeturamente i documenti dominantiche appaiono in piu cluster di alto rango ci aspettiamo che QE possa portare a risultati piu precisi.
				 la motivazione dell'utilizzo di cluster deriva dal fatto che i documenti inposizione piu` alta hanno un ordinamento "query-oriented" che nono considera la relazione tra documenti.
				 Il problema post in precedenza e` come scegliere i termini di espansione, che devono essere vicini alla quey. Selezionando ripetutamente i documenti dominanti risolviamo questa difficolta.
				 """)

st.title("Funzionamento")
st.write("""
*** Documento dominante *** : Assumiamo un documento dominanto per una query quel documento con un ottima rappresentazione dei topics di una qury, ovvero uno con diversi vicini con alta similarita`. Nel nostro caso di cluster sovrapposti un documento dominante sara` quello che appare in molteplici cluster con rango elevato. Dal documento dominante rilevato possiamo ricavare i temmini di espansioneche reperirira documenti rletivi a tutti i sottitopics.
		1. I documenti vengon o riperiti data una query usando l'API search() di elasticsearch che di base usa il BM25 senza QE
		2. Vengono generati dei cluster di k-nearest vicini per i primi N docuimenti reperiti pr trovare documenti dominanti. 
		3. In k-NN ogni docunmento hja un ruolo centrale nel creare il proprio cluster con k vicini per similarita.
		4. Il ranog viene deciso da un language model basato su cluster.
		5. I termini id espansione vengono selezionati usando il modello di rilevanza per ogni documento nei top-ranked cluster.			 
""")

st.title("Implementazone")

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

st.title("Risultati e parametri")
st.text("""Ci sono stati diversi parametri da ottimizzare, di seguito si riportano i due piu rilevanti
La prima configurazione consiste nel Reperire 10 documenti dominanti, selezionare 25 descrittori per QE, impostare il peso dei nuovi descrittore a 0.7
La seconda usa ancora 10 doicumenti dominanti, prende 100 descrittori e imposta il peso a 0.9

	
""")

st.text("""Si nota che nel training set la ocnfigurazione 10_25_07 aumenta la map, mentre sempre nel training set il modello con 10_100_09 massimizza la P@5.""")
st.text("""Non vengono inseriti i risultati del training in quanto i documenti generati da trec-eval sono tanti, Vengono pero inseriti i risultati del test.""")


				



bm25 = pl.read_csv("./Data/Eval_Test/test_results_bm25_eval.txt", has_header= False, separator="\t")

kn2507 = pl.read_csv("./Data/Eval_Test/test_results_10_25_07_eval.txt", has_header= False, separator="\t")
kn10009 = pl.read_csv("./Data/Eval_Test/test_results_10_100_09_eval.txt", has_header= False, separator="\t")

datibm = bm25.filter((pl.col("column_1").str.starts_with("map")) | (pl.col("column_1").str.starts_with("P_5 ")))
dati07 = kn2507.filter((pl.col("column_1").str.starts_with("map")) | (pl.col("column_1").str.starts_with("P_5 ")))
dati09 = kn10009.filter((pl.col("column_1").str.starts_with("map")) | (pl.col("column_1").str.starts_with("P_5 ")))

datibm[0,0] = "map_BM25"
datibm[1,0] = "P@5_BM25"

dati07[0,0] = "map_07"
dati07[1,0] = "P@5_07"
dati09[0,0] = "map_09"
dati09[1,0] = "P@5_09"

df_unito = pl.concat([datibm, dati07, dati09])


# Creare il grafico per p5~query
chart_p5 = alt.Chart(df_unito).mark_bar().encode(
  	x=alt.X('column_1', title = "Configurazione considerata"),  # Impostazione dei limiti per l'asse x
    y=alt.Y('column_3', title = "valore precisione"),
		color = 'column_1'		
).properties(
	title = "Grafico a barre precisione ~ configurazione usata (The higher the better)"
)

# Visualizzare il grafico
st.altair_chart(chart_p5, use_container_width= True)

st.text("""
Notiamo dai grafici quanto riferito in precedenza, la p@5 e maggiore con 09, mentre map e maggior econ 07. 
				""")

st.text("""
Notiamo inoltre che entrambe le scelte dei parametri migliorano sensibilmente le metriche di precisione. Quindi potremmo da subito dire che QE con clustering funziona. 
Prima pero vediamo come si comportano le query singolarmente.
				""")



st.title("Performance delle singole query P@5")
st.text("""
				Valutiamo come le singole query vengono influenzate dall\'uso dei vari metodi e dai parametri usati""")


querybm25 = pl.read_csv("./Data/Eval_Queries/bm25_evalQ.txt", has_header= True)

query07 = pl.read_csv("./Data/Eval_Queries/10_25_07_evalQ.txt", has_header= True)
query09 = pl.read_csv("./Data/Eval_Queries/10_100_09_evalQ.txt", has_header= True)
# Creare il grafico per p5~query


min_x = st.slider("Min x", min_value=600, max_value = 700, value = 1)
max_x = st.slider("Max x", min_value=600, max_value = 700, value = 1)


chart_bm25 = alt.Chart(querybm25[(querybm25['id_Q'] >= min_x) & (querybm25['id_Q'] <= max_x)]).mark_line(color = '#FF0000', filled = False).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=[600, 700])),  # Impostazione dei limiti per l'asse x
    y='p_5'
)



#chart_07 = alt.Chart(query07).mark_line(color = '#00FF00', filled = False).encode(
#  	x=alt.X('id_Q', scale=alt.Scale(domain=[600, 700])),  # Impostazione dei limiti per l'asse x
#    y='p_5'
#)
#
#chart_09 = alt.Chart(query09).mark_line(color = '#0000FF ', filled = False).encode(
#  	x=alt.X('id_Q', scale=alt.Scale(domain=[600, 700])),  # Impostazione dei limiti per l'asse x
#    y='p_5'
#)


 

# Visualizzare il grafico
st.altair_chart(chart_bm25, use_container_width= True)

st.text("""
Il grafico non e molto suggestivo, pero possiamo notare che alcune query rimangono invariate, altre invece cambiano sensibilmente""" )