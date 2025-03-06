import streamlit as st
import scipy as sci
import polars as pl


st.markdown("# Conclusione")

st.markdown("""
Abbiamo visto come ci sia una certa differenza tra le metodologie usate. 
Inoltre, tornando alla prima analisi, quella che considera globalmente **map** e **P@5** QE sembra migliorare le metriche. 
Ma le differenze sono significative?            
""")


st.markdown("""## Test per significativit\u00e0""")
st. markdown("""I test che useremo sono *Paired t-test* e *Wilcoxon signed-rank test*. Entrambi servono a confrontare due gruppi di dati correlati ma si basano su assunti diversi.
- Paired t-test \u00e8 un test che assume normalit\u00e0 dei dati;
- Wilcoxon non fa assunzione parametriche sulla distribuzione dei dati.
             
Per entrambi i testi l'ipotesi nulla corrisponde a $H_0 = $ medie uguali.
Li useremo entrambi e proveremo a trarre delle conclusioni, fissiamo un valore $\\alpha$ = 0.05
Si riportano i $p_{value}$ e le conclusioni dei test eseguiti:
""")


bm25 = pl.read_csv("./Data/Eval_Queries/bm25_evalQ.txt", has_header= True)
p5 = pl.read_csv("./Data/Eval_Queries/10_100_09_evalQ.txt", has_header= True)
map = pl.read_csv("./Data/Eval_Queries/10_25_07_evalQ.txt", has_header= True)

st.markdown("## Analisi map")

st.markdown("#### Paired t-test")
st.markdown(f"""
- configurazione che migliora P@5: {round(sci.stats.ttest_rel(bm25[:,2], p5[:,2])[1],2)}. Non rifiuto $H_0$;
- configurazione che migliora map: {round(sci.stats.ttest_rel(bm25[:,2], map[:,2])[1],2)}. Rifiuto $H_0$.
""")


st.markdown("#### Wilcoxon")

st.markdown(f"""
- configurazione che migliora P@5: {round(sci.stats.wilcoxon(bm25[:,2], p5[:,2])[1],2)}. Rifiuto $H_0$;
- configurazione che migliora map: {round(sci.stats.wilcoxon(bm25[:,2], map[:,2])[1],2)}. Rifiuto $H_0$.
""")


st.markdown("## Analisi P@5")

st.markdown("#### Paired t-test")
st.markdown(f"""
- configurazione che migliora P@5: {round(sci.stats.ttest_rel(bm25[:,1],  p5[:,1])[1],2)}. Non rifiuto $H_0$;
- configurazione che migliora map: {round(sci.stats.ttest_rel(bm25[:,1], map[:,1])[1],2)}. Non rifiuto $H_0$.
""")


st.markdown("#### Wilcoxon")

st.markdown(f"""
- configurazione che migliora P@5: {round(sci.stats.wilcoxon(bm25[:,1],  p5[:,1])[1],2)}. Non rifiuto $H_0$;
- configurazione che migliora map: {round(sci.stats.wilcoxon(bm25[:,1], map[:,1])[1],2)}. Non rifiuto $H_0$.
""")

st.markdown("""## Conclusione""")
st.markdown("""Quindi possiamo dire che l'unico valore significativo \u00e8 quello che confronta le map tra modello con QE che massimizza il map \u00e8 il modello senza QE.""")

st.markdown("""Potremmo concludere che dicendo che QE con k-nn migliora la precisione, almeno la mean average precisio, per il reperimento di documenti rilevanti. 
Se consideriamo come unica e sola metrica map, \u00e8 un metodo da scegliere, ma;
- per il training si \u00e8 semplicemente diviso il dataset in prime 150 come training e ultime 100 come test. Avendo quindi un dataset limitato non \u00e8 detto che sia la soluzione migliore. Implementare la cross validation sarebbe preferibile; 
- sempre per la fase di training e vista la natura del problema non \u00e8 stato possibile analizzare ogni combinazione possibile per i parametri $\\lambda$, $e$, ed $N$, \u00e8 quinid probabile che non siano stati considerati parametri migliori;
- i tempi di esecuzione non sono spaventosamente lunghi ma le operazioni che fa il calcolatore aumenta, aumentandone cosi l'inefficienza.
Questi sono problemi che possono essere risolti con del tempo a disposizione per il training del modello e magari considerare alcuni parametri per esempio $\\lambda$ fuori dall'insieme da considerare e trattarlo invece come parametro scelto dal programmatore e tenuto fisso per ogni configurazione.""")