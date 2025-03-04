import streamlit as st
import scipy as sci
import polars as pl


st.markdown("# Conclusione")

st.markdown("""
Abbiamo visto come ci sia una certa differenza tra le metodologie usate. 
Inoltre, tornando alla prima anali, quella che considera globalmente $map$ e $P@5$ QE sembra migliorare le metriche. 
Ma le differenze sono significative?            
""")


st.markdown("""## Test per significativita`""")
st. markdown("""I test che useremo sono *Paired t-test* e *Wilcoxon signed-rank test*. Entrambi servono a confrontare due gruppi di dati correlati ma si basano su assunti diversi.
             - Paired t-test e` un test che assume normlita` dei dati;
             - Wilcoxon non fa assunzione parametriche sulla distribuzione dei dati
             Li useremo entrambi e proveremo a trarre delle conclusioni
""")


bm25 = pl.read_csv("./Data/Eval_Queries/bm25_evalQ.txt", has_header= True)
p5 = pl.read_csv("./Data/Eval_Queries/10_100_09_evalQ.txt", has_header= True)
map = pl.read_csv("./Data/Eval_Queries/10_25_07_evalQ.txt", has_header= True)

st.markdown("## Analisi $map$")
print(bm25)
st.markdown(f"""Il risultato del test Paired t-test e {round(sci.stats.ttest_rel(bm25[:,2], p5[:,2])[1],2)}, che a un livello di significativita` $alpha$ = 0.05, viene accettata ipotesi nulla ovvero le medie non shanno differenza""")
st.markdown(f"""Il risultato del test Paired t-test e {round(sci.stats.ttest_rel(bm25[:,2], map[:,2])[1],2)}, che a un livello di significativita` $alpha$ = 0.05, viene accettata ipotesi nulla ovvero le medie non shanno differenza""")

st.markdown("## Analisi $p@5$")
print(bm25)
st.markdown(f"""Il risultato del test Paired t-test e {round(sci.stats.ttest_rel(bm25[:,1], p5[:,1])[1],2)}, che a un livello di significativita` $alpha$ = 0.05, viene accettata ipotesi nulla ovvero le medie non shanno differenza""")
st.markdown(f"""Il risultato del test Paired t-test e {round(sci.stats.ttest_rel(bm25[:,1], map[:,1])[1],2)}, che a un livello di significativita` $alpha$ = 0.05, viene accettata ipotesi nulla ovvero le medie non shanno differenza""")

st.markdown("""## Conclusione""")
st.markdown("""QUindi possiamo dire che l'unico valore significativoe e quello che confronta le map tra modello con QE che massimizza il map e il modello senza QE.""")

st.markdown("""Potremmo concludere che dicendo che QE con k-nn migliora la precisione, almeno la mean average precisio, per il reperimento di documenti rilevanti. 
            Se consideriamo come unica e sola metrica map, e` un metodo da scegliere, ma;
            - per il training si e ` semplicemente diviso il dataset in prime 150 come training e ultime 100 come test. Avendo quindi un dataset limitato non e` detto che sia la soluzione migliore. Implementare la cross validation sarebbe preferibile 
            - sempre per la fase di training e vista la natura del problema non e` stato possibile analizzare ogni combinazione possibile per i parametri $lambda$, $e$,ed $N$, e` quinid probabile che non siano stati considerati parametri migliori
            - i tempi di esecuzione non sono spaventosamente lunghi ma le operazioni che fa il calcolatore aumenta, aumentandone cosi l'inefficienza.
Questi sono problemi che possono essere risolti con del tempo a disposizione per il training del modello e magari considerare alcuni parametri per esempio $lambda$ fuori dall'insieme da considerare e trattarlo invece come parametro scelto dal programmatore e tenuto fisso per ogni configuarzione.""")

