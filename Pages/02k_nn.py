import altair as alt
import polars as pl
import streamlit as st

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
L'approccio che stiamo usando \u00E8 simile alla pseudo-relevance feedback. Nella pseudo-relevance feedback si assume che i documenti in posizione pi\u00F9 alta come rilevanti.
L'idea quindi cambia, usiamo un cluster di documenti per trovare documenti dominanti per il set reperito iniziale e ripetutamente fornire i documenti per enfatizzare gli argomenti principali di una query. 
""")

st.markdown("## Problema")
st.markdown("""
I documenti reperiti in posizione alta contengono rumore, se la P@10 \u00E8 0.5, di 10 documenti, 5 sono non rilevanti.
Il che puo provocare un drift dei risultati delle query che invece di avvicinarsi a documenti rilevanti si puo avvicinare a quelli non rilevanti.
Trovare il cluster ottimale \u00e8 difficile, quindi useremo una serie di gruppi rilevanti per il feedback permettendo cluster sovrapponibili per i top-retrieved e alimentando ripeturamente i documenti dominantiche appaiono in pi\u00F9 cluster di alto rango ci aspettiamo che QE possa portare a risultati pi\u00F9 precisi.
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
st.markdown("Mostriamo come variano le metriche scelte per l'analisi, cerchiamo in particolare se esiste una configurazione maggiormente predisposta per migliorare **map** o **P@5**")

#Carico i dati relativi ai risultati complessivi
bm25 = pl.read_csv("./Data/Eval_Test/test_results_bm25_eval.txt", has_header= False, separator="\t")
kn2507 = pl.read_csv("./Data/Eval_Test/test_results_10_25_07_eval.txt", has_header= False, separator="\t")
kn10009 = pl.read_csv("./Data/Eval_Test/test_results_10_100_09_eval.txt", has_header= False, separator="\t")

#Seleziono le metriche che mi interessano
datibm = bm25.filter((pl.col("column_1").str.starts_with("map"))    |(pl.col("column_1").str.starts_with("P_5 ")))
dati07 = kn2507.filter((pl.col("column_1").str.starts_with("map"))  |(pl.col("column_1").str.starts_with("P_5 ")))
dati09 = kn10009.filter((pl.col("column_1").str.starts_with("map")) |(pl.col("column_1").str.starts_with("P_5 ")))

datibm[0,1] = "BM25"
datibm[1,1] = "BM25"
dati07[0,1] = "first"
dati07[1,1] = "first"
dati09[0,1] = "second"
dati09[1,1] = "second"

df_unito = pl.concat([datibm, dati07, dati09])
print(df_unito)
df = pl.DataFrame({
  'method' : ["BM25", "first", "second"],
  'map' : [datibm[0,2], dati07[0,2], dati09[0,2]],
  'p5' : [datibm[1,2], dati07[1,2], dati09[1,2]]
})

chart = alt.Chart(df).mark_circle(size=400).encode(
    x='map',
    y='p5',
    color='method',
    tooltip=['map', 'p5', 'method']
).properties(
    title='Scatter plot p@5 ~ map per le configurazioni usate'
)
st.altair_chart(chart, use_container_width=True)

st.markdown("""Il grafico a dispersione mostra dei risultati interessanti. In particolare, il miglior risultato si trova nell'angolo in basso a destra, indicando una configurazione ottimale.
            
- Il BM25 si posiziona nella parte superiore sinistra del grafico, lontano dalle altre configurazioni. Questo suggerisce che, nel nostro contesto, il BM25 non rappresenta la soluzione migliore rispetto alle altre opzioni esplorate.
- La prima configurazione mostra un miglioramento del MAP, ma a discapito della P@5, indicando che si sta ottenendo una maggiore rilevanza complessiva, ma con un'accuratezza inferiore nei primi 5 risultati.
- La seconda configurazione, al contrario, aumenta la P@5, migliorando la precisione nei primi 5 risultati, ma a discapito del MAP, suggerendo che il sistema \u00e8 pi\u00F9 preciso nelle prime posizioni, ma potrebbe sacrificare la qualità complessiva dei risultati.

Nel complesso, entrambe le configurazioni sembrano portare a un miglioramento delle metriche, offrendo alcune speranze che il nuovo metodo possa effettivamente funzionare e produrre risultati migliori rispetto al BM25.""")



#Seleziono le metriche che mi interessano
datibm = bm25.filter((pl.col("column_1").str.starts_with("map"))    )
dati07 = kn2507.filter((pl.col("column_1").str.starts_with("map"))  )
dati09 = kn10009.filter((pl.col("column_1").str.starts_with("map")) )

# Rinomino la metrica in modo da capire che metodo e` stato scelto
datibm[0,0] = "BM25"
dati07[0,0] = "first"
dati09[0,0] = "second"

#Unisco i 3 dataframe in uno unico cosi da fare il grafico
df_unito = pl.concat([datibm, dati07, dati09])

df_unito = df_unito.rename({"column_1" : "configurazione"})
#Creo il grafico che mostra l'andamento delle metriche in base alla configurazione usata.
chart_map = alt.Chart(df_unito).mark_bar().encode(
  	x=alt.X('configurazione', title = "Configurazione considerata"),  # Impostazione dei limiti per l'asse x
    y=alt.Y('column_3', title = "valore map"),
		color = 'configurazione'		
).properties(
	title = "map ~ configurazione usata (The higher the better)"
)

st.altair_chart(chart_map, use_container_width= True)
st.markdown("""Vediamo che tra le configurazioni la prima, **N** = 10, **e** = 25, **$\\lambda$** = 0.7, migliora sensibilmente il **map**""")



#Seleziono le metriche che mi interessano
datibm = bm25.filter((pl.col("column_1").str.starts_with("P_5 "))    )
dati07 = kn2507.filter((pl.col("column_1").str.starts_with("P_5 "))  )
dati09 = kn10009.filter((pl.col("column_1").str.starts_with("P_5 ")) )

# Rinomino la metrica in modo da capire che metodo e` stato scelto
datibm[0,0] = "BM25"
dati07[0,0] = "first"
dati09[0,0] = "second"

#Unisco i 3 dataframe in uno unico cosi da fare il grafico
df_unito = pl.concat([datibm, dati07, dati09])

df_unito = df_unito.rename({"column_1" : "configurazione"})
#Creo il grafico che mostra l'andamento delle metriche in base alla configurazione usata.
chart_p5 = alt.Chart(df_unito).mark_bar().encode(
  	x=alt.X('configurazione', title = "Configurazione considerata"),  # Impostazione dei limiti per l'asse x
    y=alt.Y('column_3', title = "valore P@5"),
		color = 'configurazione'		
).properties(
	title = "P@5 ~ configurazione usata (The higher the better)"
)
st.altair_chart(chart_p5, use_container_width= True)

st.markdown("""Vediamo che tra le configurazioni la seconda, **N** = 10, **e** = 100, **$\\lambda$** = 0.9, migliora sensibilmente la P@5""")
st.markdown(""""In seguito, analizzeremo in dettaglio le due configurazioni, esaminando come le performance relative alle query vengano modificate in base agli aspetti che risultano maggiormente migliorati. Nel caso della prima configurazione, considereremo la metrica **map**, mentre nella seconda ci concentreremo sulla metrica **P@5**. """)



st.markdown("# Modello che massimizza P@5")
st.markdown("""Prendiamo in analisi il modello che massimizza la metrica P@5, ovvero quello con parametri $lambda = 0.9$, $e = 100$ ed $N$ = 10. 
Valutiamo come le singole query vengono influenzate dall'uso di questo modello e quali migliorano o peggiorano""")

#carico dati relativi alle singole query
querybm25 = pl.read_csv("./Data/Eval_Queries/bm25_evalQ.txt", has_header= True)
query09 = pl.read_csv("./Data/Eval_Queries/10_100_09_evalQ.txt", has_header= True)

#Creo slider per selezionare le query che mi interessano
selected_x = st.slider("Seleziona il valore di query per p@5", min_value=MIN_SLIDE, max_value=MAX_SLIDE, value=(MIN_SLIDE, MAX_SLIDE))
#Filtro le query in bse allo slider
filtered_bm = querybm25.filter((querybm25['id_Q'] >= selected_x[0]) & (querybm25['id_Q'] <= selected_x[1]))
filtered_09 = query09.filter((query09['id_Q'] >= selected_x[0]) & (query09['id_Q'] <= selected_x[1]))

#creo grafico che mostra andamento della metrica p@5 usando il metodo senza QE
chart_bm = alt.Chart(filtered_bm).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='p_5', 
		color=alt.Color('method', 
                    scale=alt.Scale(range=['blue', 'yellow']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)
#creo grafico che mostra andamento della metrica p@5 usando il metodo con QE che massimizza P@5
chart_09 = alt.Chart(filtered_09).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='p_5',
		color=alt.Color('method', 
                    scale=alt.Scale(range=['blue', 'yellow']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)

st.altair_chart(chart_bm+chart_09, use_container_width= True)

st.text("""
Il grafico non \u00e8 molto suggestivo, per\u00f2 possiamo notare che alcune query rimangono invariate, altre invece cambiano sensibilmente.
""" )


# Individuo le query che migliorano e quali peggiorano maggiormente
q_25 = querybm25
q_09 = query09
q_09_merge = q_25.join(q_09, on ="id_Q")
q_09_merge= q_09_merge.with_columns(
    (pl.col("p_5_right") - pl.col("p_5")).alias("difference")
)


# Calcola la differenza massima e minima
max_diff = q_09_merge["difference"].max()
min_diff = q_09_merge["difference"].min()

# Trovo i punti di massimo e minimo
max_index = q_09_merge["difference"].arg_max()
min_index = q_09_merge["difference"].arg_min()

st.markdown(f"""### Prendiamo per esempio le seguenti query:
- {max_index + OFFSET}: P@5 senzas QE di {querybm25[max_index,1]} mentre usando QE otteniamo {query09[max_index,1]}. Il nostro meteodo ha funzionato egregiamente, anche se questo \u00E8 un caso di esempio dove viene massimizzata la differenza.
- {min_index + OFFSET}: P@5 senza QE di {querybm25[min_index,1]}, mentre usando QE P@5 diventa di {query09[min_index,1]}. Ovvero viene dimezzata la precisione. 
""")


st.markdown("# Modello che massimizza map")
st.markdown("""Prendiamo in analisi il modello che massimizza la metrica map, ovvero quello con parametri $lambda = 0.7$, $e = 25$ ed $N$ = 10. 
Valutiamo come le singole query vengono influenzate dall'uso di questo modello e quali migliorano o peggiorano""")

#carico il dataset relativo alle singole query del modello che massimizza map
query07 = pl.read_csv("./Data/Eval_Queries/10_25_07_evalQ.txt", has_header= True)

#Creo slider per selezioanre query di interesse
selected_x = st.slider("Seleziona il valore di query per map", min_value=MIN_SLIDE, max_value=MAX_SLIDE, value=(MIN_SLIDE, MAX_SLIDE))
#Filtor le query inbase alle regole dello slider
filtered_bm = querybm25.filter((querybm25['id_Q'] >= selected_x[0]) & (querybm25['id_Q'] <= selected_x[1]))
filtered_07 = query07.filter((query07['id_Q'] >= selected_x[0]) & (query07['id_Q'] <= selected_x[1]))

#creo grafico che mostra andamento della metrica mpa usando il metodo senza QE
chart_bm = alt.Chart(filtered_bm).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='map', 
		color=alt.Color('method', 
                    scale=alt.Scale(range=['blue', 'yellow']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)
#creo grafico che mostra andamento della metrica map usando il metodo con QE che massimizza map
chart_07 = alt.Chart(filtered_07).mark_point(filled = True).encode(
  	x=alt.X('id_Q', scale=alt.Scale(domain=selected_x)),  # Impostazione dei limiti per l'asse x
    y='map',
		color=alt.Color('method', 
                    scale=alt.Scale(range=['blue', 'yellow']),  # Imposta i colori desiderati
                    legend=alt.Legend(title='Method'))
)

st.altair_chart(chart_bm+chart_07, use_container_width= True)


st.text("""
Il grafico non \u00e8 molto suggestivo, per\u00F2 possiamo notare che alcune query rimangono invariate, altre invece cambiano sensibilmente.
""" )

# Individuo le query che migliorano e quali peggiorano maggiormente
q_07 = query07
q_07_merge = q_25.join(q_07, on ="id_Q")
q_07_merge= q_07_merge.with_columns(
    (pl.col("map_right") - pl.col("map")).alias("difference")
)

# Calcola la differenza massima e minima
max_diff = q_07_merge["difference"].max()
min_diff = q_07_merge["difference"].min()

# Trovo i punti di massimo e minimo
max_index = q_07_merge["difference"].arg_max()
min_index = q_07_merge["difference"].arg_min()

st.markdown(f"""### Prendiamo per esempio le seguenti query:
- {max_index + OFFSET}: map senza QE di {querybm25[max_index,2]} mentre usando QE otteniamo {query09[max_index,2]}. Il nostro meteodo ha funzionato egregiamente, anche se questo \u00E8 un caso di esempio dove viene massimizzata la differenza.
- {min_index + OFFSET}: map senza QE di {querybm25[min_index,2]}, mentre usando QE mapdiventa di {query09[min_index,2]}. meno della meta della precisione. 
""")


st.markdown("""Si possono fare ulteriori considerazioni sul perch\u00e9 alcune query hanno funzionato meglio di altre ma non \u00E8 l'obbiettivo di questo progetto.""")