
#installazione ed import librerie
!pip install pyspark
!pip install sparknlp
import sparknlp
from pyspark.sql import SparkSession

#creazione ambiente spark
spark = sparknlp.start(gpu=True) #gpu=True solo se macchina possiede GPU

#lettura del file json contenente il dataset (lettura con spark)
df = spark.read.option("header", "true").json("/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json")
'''df.show(5)
numero_articoli_dataset = df.count()
print(f"Ci sono {numero_articoli_dataset} articoli nel dataset")'''

#stampa delle prime righe del dataset
df.head()

#cancellazione righe non utili per test con BERT
drop_list = ['id', 'submitter', 'authors', 'title', 'comments', 'journal-ref', 'doi', 'report-no', 'license', 'versions', 'update_date', 'authors_parsed']
df2 = df.select([column for column in df.columns if column not in drop_list])
df2.show(5)

#import librerie per suddividere la colonna category (ogni colonna categoria - e quindi ogni articolo può avere piu di una categoria)
from os.path import isfile, join

import pandas as pd
import numpy as np
from pyspark.sql.functions import split
from pyspark.sql.functions import when, col


df3 = df2.withColumn('category1', split(df2['categories'], ' ').getItem(0)) \
        .withColumn('category2', split(df2['categories'], ' ').getItem(1)) \
        .withColumn('category3', split(df2['categories'], ' ').getItem(2)) \
        .withColumn('category4', split(df2['categories'], ' ').getItem(3)) \
        .withColumn('category5', split(df2['categories'], ' ').getItem(4))

#creazione e filtaggio con un flag che mi indica tutti quegli articoli che hanno almeno una categoria facente parte della macrocategoria CS)
df4 = df3.withColumn("abstract_with_or_without_category_cs", when(df3["category1"].rlike('^cs\.'), 1). when(df3["category2"].rlike('^cs\.'), 1). when(df3["category3"].rlike('^cs\.'), 1) . when(df3["category4"].rlike('^cs\.'), 1). when(df3["category5"].rlike('^cs\.'), 1). otherwise(0))
df5 = df4.filter(col("abstract_with_or_without_category_cs") == 1)

#cancellazione delle categorie non utili per la singola run del codice - ogni volta scambiare le categoria - una rimane e le altre vengono cancellate)
drop_list2 = ['categories', 'category4', 'category3', 'category2', 'category5', 'abstract_with_or_without_category_cs']
df6 = df5.select([column for column in df5.columns if column not in drop_list2])
df7 = df6.filter(col("category1").rlike('^cs\.'))
df7.show()

#suddivisione del dataset per permettere a BERT di runnare in un tempo congruo (15-50 minuti)
meta1, meta2, meta3, meta4 = df7.randomSplit([0.25, 0.25, 0.25, 0.25], seed=42)

#suddivisione in trainset e in testset della parte di dataset presa in esame
trainDataset, testDataset = meta4.randomSplit([0.7, 0.3], seed=42)

#conteggio delle righe facenti parte del trainset e del testset
print(trainDataset.count())
print(testDataset.count())

#import delle librerie per far runnare BERT
from sparknlp.annotator import *
from sparknlp.base import *

#settaggio del modello BERT
document = DocumentAssembler()\
    .setInputCol("abstract")\
    .setOutputCol("document")
    
bert_sent = BertSentenceEmbeddings.pretrained("sent_small_bert_L8_512")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")
classsifierdl = ClassifierDLApproach()\
 .setInputCols(["sentence_embeddings"])\
 .setOutputCol("class")\
 .setLabelColumn("category1")\
 .setMaxEpochs(1000)\
 .setEnableOutputLogs(True)
bert_sent_clf_pipeline = Pipeline(
    stages = [
        document,
        bert_sent,
        classsifierdl
])

#utilizzo del modello BERT con i settaggi sopra indicati
use_pipelineModel = bert_sent_clf_pipeline.fit(trainDataset)

#import delle librerie per la visualizzazione dei risultati finali
!pip install sklearn
from sklearn.metrics import classification_report, accuracy_score

#test sul testset con il train che il modello ha effetuato sul trainset
df8 = use_pipelineModel.transform(testDataset).select('category1', 'abstract', "class.result").toPandas()
df8['result'] = df8['result'].apply(lambda x: x[0])

#visualizzazione dei risultati finali
print(classification_report(df8.category1, df8.result))
print(accuracy_score(df8.category1, df8.result))