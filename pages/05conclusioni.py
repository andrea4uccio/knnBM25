import streamlit as st
import scipy as sci
import polars as pl

N_DIGIT = 4
st.set_page_config(
    page_title="conclusioni"
)

st.markdown("# Conclusione")

st.markdown("""
Abbiamo visto come ci sia una certa differenza tra le metodologie usate. 
Il metodo che implementa la ricerca dei k-vicini peggiora le metriche considerate mentre il metodo che implementa word2vec le migliora. 
Ma le differenze sono significative?            
""")


st.markdown("""## Test per significativit\u00e0""")
st. markdown("""I test che useremo sono *Paired t-test* e *Wilcoxon signed-rank test*. Entrambi servono a confrontare due gruppi di dati correlati ma si basano su assunti diversi.
- Paired t-test \u00e8 un test che assume normalit\u00e0 dei dati;
- Wilcoxon non fa assunzione parametriche sulla distribuzione dei dati.
             
Per entrambi i testi l'ipotesi nulla corrisponde a $H_0 = $ medie uguali.
Li useremo entrambi e proveremo a trarre delle conclusioni, fissiamo un valore $\\alpha$ = 0.05
Si riportano i $p_{{value}}$ e le conclusioni dei test eseguiti:
""")

#Carico i dati relativi alle singole querty per test di significativita``
bm25 = pl.read_csv("./Data/EVAL_TEST_Q/base/Eval_Q_QE_base_combined.csv", has_header= True)
knn = pl.read_csv("./Data/EVAL_TEST_Q/knn/Eval_Q_QE_knn_combined.csv", has_header= True)
w2v = pl.read_csv("./Data/EVAL_TEST_Q/w2v/Eval_Q_QE_w2v_combined.csv", has_header= True)


st.markdown("## Analisi MAP")

st.markdown(f"""
            Partiamo con vedere le differenze in media tra la baseline e i metodi proposti. 
- Metodo che implementa k-NN, la differenza con la baseline \u00e8: {round(
      -bm25.get_column("map").mean() + knn.get_column("map").mean(), N_DIGIT )}, il metodo peggiora MAP
- MEtodo che  implementa Word2Vec, la differenza \u00e8 : {round(
      -bm25.get_column("map").mean() + w2v.get_column("map").mean(), N_DIGIT )}, il metodo migliora la MAP.   
""")
 
st.markdown("#### Paired t-test")
st.markdown(f"""
- metodo che implementa k-nn: $p_{{value}}$ = {round(sci.stats.ttest_rel(
    bm25.get_column("map"),
    knn.get_column("map"))[1],N_DIGIT)}. Non rifiuto $H_0$, ho una differenza significativa, il metodo peggiora la metrica;

- metodo che implementa Word2Vec: $p_{{value}}$ = {round(sci.stats.ttest_rel(
    bm25.get_column("map"),
    w2v.get_column("map")
  )[1],N_DIGIT)}. La differenza \u00e8 positiva, il metodo migliora in modo significativo la metrica.
""")


st.markdown("#### Wilcoxon")
st.markdown(f"""
- metodo che implementa k-nn: $p_{{value}}$ = {round(sci.stats.wilcoxon(
    bm25.get_column("map"),
    knn.get_column("map"))[1],N_DIGIT)}. Non rifiuto $H_0$, ho una differenza significativa, il metodo peggiora la metrica;

- metodo che implementa Word2Vec: $p_{{value}}$ = {round(sci.stats.wilcoxon(
    bm25.get_column("map"),
    w2v.get_column("map")
  )[1],N_DIGIT)}. La differenza \u00e8 positiva, il metodo migliora in modo significativo la metrica.
""")

st.markdown("## Analisi P@5")
st.markdown(f"""
            Partiamo con vedere le differenze in media tra la baseline e i metodi proposti. 
- Metodo che implementa k-NN, la differenza con la baseline \u00e8: {round(
      -bm25.get_column("p_5").mean() + knn.get_column("p_5").mean(), N_DIGIT )}, il metodo peggiora p_5
- MEtodo che  implementa Word2Vec, la differenza \u00e8 : {round(
      -bm25.get_column("p_5").mean() + w2v.get_column("p_5").mean(), N_DIGIT )}, il metodo migliora la p_5.   
""")
 
st.markdown("#### Paired t-test")
st.markdown(f"""
- metodo che implementa k-nn: $p_{{value}}$ = {round(sci.stats.ttest_rel(
    bm25.get_column("p_5"),
    knn.get_column("p_5"))[1],N_DIGIT)}. Non rifiuto $H_0$, ho una differenza significativa, il metodo peggiora la metrica;

- metodo che implementa Word2Vec: $p_{{value}}$ = {round(sci.stats.ttest_rel(
    bm25.get_column("p_5"),
    w2v.get_column("p_5")
  )[1],N_DIGIT)}. La differenza \u00e8 positiva ma non significativa.
""")


st.markdown("#### Wilcoxon")
st.markdown(f"""
- metodo che implementa k-nn: $p_{{value}}$ = {round(sci.stats.wilcoxon(
    bm25.get_column("p_5"),
    knn.get_column("p_5"))[1],N_DIGIT)}. Non rifiuto $H_0$, ho una differenza significativa, il metodo peggiora la metrica;

- metodo che implementa Word2Vec: $p_{{value}}$ = {round(sci.stats.wilcoxon(
    bm25.get_column("p_5"),
    w2v.get_column("p_5")
  )[1],N_DIGIT)}. La differenza \u00e8 positiva, il metodo migliora la metrica ma non in modo significativo.
""")


st.markdown("## Analisi nDCG@10")
st.markdown(f"""
            Partiamo con vedere le differenze in media tra la baseline e i metodi proposti. 
- Metodo che implementa k-NN, la differenza con la baseline \u00e8: {round(
      -bm25.get_column("ndcg_10").mean() + knn.get_column("ndcg_10").mean(), N_DIGIT )}, il metodo peggiora nDCG@10
- MEtodo che  implementa Word2Vec, la differenza \u00e8 : {round(
      -bm25.get_column("ndcg_10").mean() + w2v.get_column("ndcg_10").mean(), N_DIGIT )}, il metodo migliora la nDCG@10.   
""")
 
st.markdown("#### Paired t-test")
st.markdown(f"""
- metodo che implementa k-nn: $p_{{value}}$ = {round(sci.stats.ttest_rel(
    bm25.get_column("ndcg_10"),
    knn.get_column("ndcg_10"))[1],N_DIGIT)}. Non rifiuto $H_0$, ho una differenza significativa, il metodo peggiora la metrica;

- metodo che implementa Word2Vec: $p_{{value}}$ = {round(sci.stats.ttest_rel(
    bm25.get_column("ndcg_10"),
    w2v.get_column("ndcg_10")
  )[1],N_DIGIT)}. La differenza \u00e8 positiva ma non significativa.
""")


st.markdown("#### Wilcoxon")
st.markdown(f"""
- metodo che implementa k-nn: $p_{{value}}$ = {round(sci.stats.wilcoxon(
    bm25.get_column("ndcg_10"),
    knn.get_column("ndcg_10"))[1],N_DIGIT)}. Non rifiuto $H_0$, ho una differenza significativa, il metodo peggiora la metrica;

- metodo che implementa Word2Vec: $p_{{value}}$ = {round(sci.stats.wilcoxon(
    bm25.get_column("ndcg_10"),
    w2v.get_column("ndcg_10")
  )[1],N_DIGIT)}. La differenza \u00e8 positiva, il metodo migliora la metrica ma non in modo significativo.
""")




st.markdown("""## Conclusione""")
st.markdown("""Possiamo dire che il metodo che implementa l'espansione della query utilizzando Word2Vec migliora l'efficacia del reperimento con risultati significativi per la MAP. Le altre metriche, se pur non avendo risultati statisticamente significativi, comunque migliorano. Questo dimostra che l'espansione delle query utilizzando un modello di word embedding denso migliora i risultati.""")

st.markdown("""
I risultati vanno comunque analizzati considerando quanto segue:
- per il training si \u00e8 semplicemente diviso il dataset in prime 150 come training e ultime 100 come test. Avendo quindi un dataset limitato non \u00e8 detto che sia la soluzione migliore. Implementare la cross validation sarebbe preferibile; 
- sempre per la fase di training e vista la natura del problema non \u00e8 stato possibile analizzare ogni combinazione possibile per i parametri dei metodi. Una ricerca pi\u00f2 esaustiva potrebbe rivelare che anche il metodo che implementa k-nn migliora i risultati, magari anche piu di Word2Vec;
- i tempi di esecuzione non sono spaventosamente lunghi ma le operazioni che fa il calcolatore aumenta, aumentandone cosi l'inefficienza.
            
Questi sono problemi che possono essere risolti con del tempo a disposizione per il training del modello e magari considerare alcuni parametri per esempio $\\lambda$ fuori dall'insieme da considerare e trattarlo invece come parametro scelto dal programmatore e tenuto fisso per ogni configurazione.""")