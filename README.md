
# Classification Arxiv Dataset

Di seguito troverete una semplice descrizione su due metodi di classificazione utilizzati nel dataset Arxiv.

- BERT
- Zero-shot Classification


## Acknowledgements

 - [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)


## Appendix

Ora faremo una breve descrizione dei metodi di classificazione utilizzati:

- BERT: è l’acronimo inglese di Bidirectional Encoder Representations from Transformers e consiste in un sistema per l’elaborazione del linguaggio naturale, il Natural Language Processing (NLP), che si basa sul concetto di reti neurali (modelli matematici composti da neuroni artificiali che si ispirano al funzionamento biologico del cervello umano).

- Zero-Shot Classification: la classificazione del testo zero-shot è un'attività nell'elaborazione del linguaggio naturale in cui un modello viene addestrato su una serie di esempi etichettati. E' quindi in grado di classificare nuovi esempi da classi mai viste prima.


## Authors

- [Stefano Cabiddu](https://github.com/StefanoCabiddu)


## Deployment

Per utilizzare questo progetto runnare su google colab (o su un'IDE se in possesso di GPU fisica nel PC che si utilizza) il codice che si trova nei file python relativi ai metodi di classificazione utilizzati.

Nel caso si utilizzasse google colab bisogna caricare il dataset sul blocco note su cui si lavora (tramite collegamento a google drive o tramite caricamento file - il secondo metodo è molto più lento)






## Documentation

In allegato si può trovare la documentazione relativa ai risultati ottenuti. 

Nel primo file, infatti, si può trovare una guida specifica sul lavoro effettuato mentre negli altri due si possono trovare i risultati ottenuti dai due metodi di classificazione utilizzati. 

[Documentazione risultati ottenuti](https://drive.google.com/file/d/1mj2jYzGD_Aj3QKqIdGcL0abTpqgqPUvF/view?usp=sharing)

[Risultati classificazione tramite BERT](https://drive.google.com/file/d/18sbAe56YCVRXAr-8ScGuo8SI4EGZ_bRe/view?usp=sharing)

[Risultati classificazione tramite Zero-shot Learning](https://drive.google.com/file/d/1wtKJxPnTf-xEKWnZ4FPBKBy593Agly2U/view?usp=sharing)

## Features

- Possibilità di eseguire il codice su Google Colab
- Codice semplice da capire
- Utilizzo di Apache Spark
- Scritto in Python


## Feedback

Per qualsiasi commento, contattatemi all'indirizzo stefcabiddu@gmail.com.


## License

Il codice sorgente è stato rilasciato sotto la licenza open source Apache 2.0.

## Optimizations

Ho ritenuto utile nello Zero-shot Learning passare il dataframe creato con Apache Spark a Pandas per ottimizzare quella che è la classificazione del modello. Infatti, come ho spiegato anche nella documentazione che si può scaricare da questo readme nella sezione "Documentation", il metodo che avrei dovuto utilizzare per recuperare tutti gli elementi del set di dati (da tutti i nodi) nel nodo driver è denominato collect(). Esso, però, per la grande mole di dati su cui doveva lavorare aveva un tempo di esecuzione stimato in parecchie ore (si arrivava anche a 16 ore per il suo completamento)

Per questo motivo è stato appunto deciso di trasformare il dataframe in pandas in modo tale che utilizzando il metodo iloc l'esecuzione della classificazione poteva avvenire in una tempistica accettabile (massimo 45/50 minuti per le porzioni dataset più grandi). 

## Lessons Learned

Le sfide che questo progetto ha dato da superare sono molteplici. Innanzitutto si sono cercati e trovati metodi e modi per ottimizzare il codice quanto più possibile.

C'è stata sicuramente una ricerca ed uno studio per quanto riguarda i metodi di classificazione da utilizzare. Lo studio ha riguardato anche la ricerca del codice più adatto per ottenere i risultati migliori. 

Un'altra sfida a cui si è trovata soluzione è stat quella di permettere a chi non possedesse una GPU su cui poter far eseguire il codice di utilizzare lo stesso. La soluzione trovata è stata quella di esportare il codice su Google Colab. I test sono stati fatti su quella piattaforma e possono essere visibili nella documentazione riportata sopra.
## Roadmap

- Browser support (Google Colab)
- Add more integrations


## Support

Per ricevere assistenza, inviate un'e-mail a stefcabiddu@gmail.com


## Tech Stack

**Client:** Python

**Server:** Apache Spark, Pandas, Scikit-learn

## Run Locally

Per poter utilizzare questo progetto ed ottenere i risultati richiesti è possibile scaricare il codice sorgente da questo repository github e farlo eseguire o su Google Colab oppure su un IDE installato sul proprio PC. 

Nel caso si volesse utilizzare il primo metodo ecco che è utile caricare (meglio tramite Google Drive) il dataset ArXiv. Un upload classico, infatti, essendo il dataset molto pesante in termini di spazio, ha bisogno di tanto tempo (si parla di ore) per la propria esecuzione.