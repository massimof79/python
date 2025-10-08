

#Inserimento dei voti

print("Quanti voti vuoi inserire?")
n = int(input())
voti = []   #lista vuota
for i in range(n):
    print("Inserisci il voto numero", i+1)
    print("Inserisci il nome dello studente:")
    nome = input()
    print("Inserisci il cognome dello studente:")
    cognome = input()
    print("Inserisci il voto dello studente:")
    voto = int(input())
    print("Inserisci la materia a cui si riferisce il voto:")
    materia = input()
    voti.append((nome, cognome, voto, materia))  #aggiungo una tupla alla lista

print(voti)
#Calcolo della media dei voti
somma = 0
for i in range(n):
    somma = somma + voti[i][2]  #voti[i] è la tupla, voti[i][2] è il voto
media = somma / n
print("La media dei voti è:", media)    

#Calcolo della media dei voti per materia
materie = []  #lista vuota per le materie
for i in range(n):
    if voti[i][3] not in materie:  #se la materia non è già nella lista
        materie.append(voti[i][3])  #aggiungo la materia alla lista delle materie   
print("Le materie sono:", materie)

for materia in materie:
    somma = 0
    count = 0
    for i in range(n):
        if voti[i][3] == materia:  #se la materia corrisponde
            somma = somma + voti[i][2]  #aggiungo il voto alla somma
            count = count + 1  #incremento il contatore
    media = somma / count
    print("La media dei voti per la materia", materia, "è:", media)

#Stampo tutte le informazioni di ogni studente
for i in range(n):
    print("Studente:", voti[i][0], voti[i][1], "Voto:", voti[i][2], "Materia:", voti[i][3]) 
