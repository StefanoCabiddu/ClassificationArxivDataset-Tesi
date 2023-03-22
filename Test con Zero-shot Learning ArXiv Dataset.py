#installazione delle librerie su Google Colab (se si utilizza UN IDE installare le librerie per come richiesto)
!pip install pyspark
!pip install sparknlp
!pip install -q git+https://github.com/huggingface/transformers.git

#import delle librerie da utilizzare
from transformers import pipeline
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import sparknlp
from pyspark.sql import SparkSession

#creazione e utilizzo dell'ambiente SPARK
spark = sparknlp.start(gpu=True) #gpu=True solo se macchina possiede GPU

df = spark.read.option("header", "true").json("/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json")

#mappatura delle categorie da utilizzare per lo Zero-shot Learning
category_map = {
                'cs.AI': 'Artificial Intelligence',
                'cs.AR': 'Hardware Architecture',
                'cs.CC': 'Computational Complexity',
                'cs.CE': 'Computational Engineering, Finance, and Science',
                'cs.CG': 'Computational Geometry',
                'cs.CL': 'Computation and Language',
                'cs.CR': 'Cryptography and Security',
                'cs.CV': 'Computer Vision and Pattern Recognition',
                'cs.CY': 'Computers and Society',
                'cs.DB': 'Databases',
                'cs.DC': 'Distributed, Parallel, and Cluster Computing',
                'cs.DL': 'Digital Libraries',
                'cs.DM': 'Discrete Mathematics',
                'cs.DS': 'Data Structures and Algorithms',
                'cs.ET': 'Emerging Technologies',
                'cs.FL': 'Formal Languages and Automata Theory',
                'cs.GL': 'General Literature',
                'cs.GR': 'Graphics',
                'cs.GT': 'Computer Science and Game Theory',
                'cs.HC': 'Human-Computer Interaction',
                'cs.IR': 'Information Retrieval',
                'cs.IT': 'Information Theory',
                'cs.LG': 'Machine Learning',
                'cs.LO': 'Logic in Computer Science',
                'cs.MA': 'Multiagent Systems',
                'cs.MM': 'Multimedia',
                'cs.MS': 'Mathematical Software',
                'cs.NA': 'Numerical Analysis',
                'cs.NE': 'Neural and Evolutionary Computing',
                'cs.NI': 'Networking and Internet Architecture',
                'cs.OH': 'Other Computer Science',
                'cs.OS': 'Operating Systems',
                'cs.PF': 'Performance',
                'cs.PL': 'Programming Languages',
                'cs.RO': 'Robotics',
                'cs.SC': 'Symbolic Computation',
                'cs.SD': 'Sound',
                'cs.SE': 'Software Engineering',
                'cs.SI': 'Social and Information Networks',
                'cs.SY': 'Systems and Control'
                }

#cancellazione delle colonne non utilizzate
drop_list = ['id', 'submitter', 'authors', 'comments', 'title', 'journal-ref', 'doi', 'report-no', 'license', 'versions', 'update_date', 'authors_parsed']
df2 = df.select([column for column in df.columns if column not in drop_list])
df2.show(5)

#import di altre librerie atte a preparare il dataset per la conseguente analisi con il metodo di classificazione
import os
from os.path import isfile, join

import pandas as pd
import numpy as np
from pyspark.sql.functions import split

#suddivisione della colonna category per permettere al metodo di classificazione di analizzare i dati 
df3 = df2.withColumn('category1', split(df2['categories'], ' ').getItem(0)) \
        .withColumn('category2', split(df2['categories'], ' ').getItem(1)) \
        .withColumn('category3', split(df2['categories'], ' ').getItem(2)) \
        .withColumn('category4', split(df2['categories'], ' ').getItem(3)) \
        .withColumn('category5', split(df2['categories'], ' ').getItem(4))

#cancellazione delle sottocategorie non utilizzate nella singola run del codice - in ogni run si andrà a modificare
#quella che è la sottocategoria utilizzata 
df4 = df3.withColumn("abstract_with_or_without_category_cs", when(df3["category1"].rlike('^cs\.'), 1). when(df3["category2"].rlike('^cs\.'), 1). when(df3["category3"].rlike('^cs\.'), 1) . when(df3["category4"].rlike('^cs\.'), 1). when(df3["category5"].rlike('^cs\.'), 1). otherwise(0))
df5 = df4.filter(col("abstract_with_or_without_category_cs") == 1)
drop_list2 = ['categories', 'category1', 'category4', 'category2', 'category3', 'abstract_with_or_without_category_cs']
df6 = df5.select([column for column in df5.columns if column not in drop_list2])
df7 = df6.filter(col("category5").rlike('^cs\.'))
df7.show()

#per le category 1 e 2 ecco che, tenendo conto delle risorse gratuitamente disponibili su Google Colab,
#si è dovuto suddividere il dataset in parti
#meta1, meta2, meta3 = df7.randomSplit([0.4, 0.3, 0.3], seed=42)# - questo è un esempio relativo alla colonna Category2

#suddivisione della porzione di dataset per permettere al metodo di classificazione Zero-shot-Learning
#di poter lavorare con lo stesso numero di righe con cui lavora il metodo di classificazione BERT
dataset_rimanente, test = df7.randomSplit([0.7, 0.3], seed=42)

#per le category 3, 4 e 5 ecco che si debba sorvolare il pezzo di codice precedente e si può ripartire da qui
#E' stato scelto di trasformare in Pandas il dataframe che si sta utilizzando proprio per l'utilizzo del metodo
#iloc nel modello di classificazione. Su Apache Spark un metodo simile a quello iloc per Pandas, è chiamato
#collect. Leggere il README del progetto per capire meglio il motivo di questa scelta.
df8 = test.toPandas()

#inizializzazione del metodo di classificazione
classifier = pipeline("zero-shot-classification",device = 0)

#utilizzo del metodo di classificazione
candidate_labels = list(category_map.values()) #le etichette candidate sono fondamentalmente le classi che il classificatore predirà.
trueCategories = []
for i in tqdm(range(20)): #si andrà a modificare per ogni run il numero contenuto all'interno del range
    text = df8.iloc[i,]['abstract']
    cat = df8.iloc[i,]['category5']#per ogni run si andrà a modificare quella che è la categoria presa in esame
    cat = cat.split()
    res = classifier(text, candidate_labels, multi_label=True)#bisogna settare il valore di Multi Label a True
    labels = res['labels'] 
    scores = res['scores'] #estrazione dei punteggi associati alle etichette
    res_dict = {label : score for label,score in zip(labels, scores)}
    sorted_dict = dict(sorted(res_dict.items(), key=lambda x:x[1],reverse = True)) #si ordina il dizionario delle etichette in ordine decrescente in base al loro punteggio
    categories  = []
    for i, (k,v) in enumerate(sorted_dict.items()):
        if(i > 0): #si memorizza solo la categoria che ha il punteggio più alto
            break
        else:
            categories.append(k)
    predictedCategories.append(categories)
    trueCats = [category_map[x] for x in cat]
    trueCategories.append(trueCats)

#stampa della categoria reale e della categoria prevista
for y_true, y_pred in zip(trueCategories[:10], predictedCategories[:10]):
    print(f'True Categories {y_true}')
    print(f'Predicted Categories {y_pred}')
    print('#'*50)

#preparazione dei risultati per studiare se il metodo di classificazione ha prodotto i migliori risultati possibili
category_idx = {cat : i for i,cat in enumerate(category_map.values())}

y_trueEncoded = []
y_predEncoded = []
for y_true, y_pred in zip(trueCategories, predictedCategories):
    encTrue = [0] * len(category_map)
    for cat in y_true:
        idx = category_idx[cat]
        encTrue[idx] = 1
    y_trueEncoded.append(encTrue)
    encPred = [0] * len(category_map)
    for cat in y_pred:
        idx = category_idx[cat]
        encPred[idx] = 1
    y_predEncoded.append(encPred)

#calcolo dell'accuratezza
from sklearn.metrics import accuracy_score
print(f'The accuracy_score is {accuracy_score(y_trueEncoded,y_predEncoded):.4f}')

#calcolo dell'Hamming Loss
from sklearn.metrics import hamming_loss
print(f'The hamming loss is {hamming_loss(y_trueEncoded,y_predEncoded):.4f}')