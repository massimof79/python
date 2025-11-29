from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
import gradio as gr

#apertura flusso per aggiungere i giocatori
file=open("archivio.txt","a")

#realizzazione della previsione
def ai(età_videogiocatore,sesso_videogiocatore): 

    previsione=modello.predict([[sesso_videogiocatore,età_videogiocatore]])    
    return "Al gamer piacciono i giochi "+ str(previsione)

#preparazione del modello (suddivisoni delle colonne del file csv in valori e risultati)
giocatori= read_csv('giocatori.csv')
X=giocatori.drop(columns=['videogame']) #drop serve per selezionare tutte le colonne meno "videogame"
y=giocatori['videogame']
#addestramento del modello
modello=DecisionTreeClassifier()
modello.fit(X.values,y.values)
#acquisizione valori
età_videogiocatore=int(input("inserisci età del video giocatore "))
sesso_videogiocatore=int(input("inserisci 0 se femmina o 1 se maschio "))
#previsione
previsione=ai(età_videogiocatore,sesso_videogiocatore)
print("Al gamer piacciono i giochi "+ str(previsione))

#scrittura su file
file.write("\n")
file.write(str(età_videogiocatore))
file.write(" ")
file.write(str(sesso_videogiocatore))
file.write(" ")
file.write(previsione)
file.close()


#interfaccia web con Gradio
demo=gr.Interface(
    fn= ai,
    inputs=['text','text'],
    outputs=[gr.Textbox(label="Gioco consigliato", lines=3)],
    title="Scelta videogiochi da regalare",
    description="Ecco un programma che ti permette di capire quale gioco regalare ai tuoi amici grazie alla loro età ed il loro sesso (0=femmina,1=maschio)",
     examples=[
        [23,0],
        [12,1],
        [44,0],
        [35,1],
    ],
)

demo.launch()
