"""Scrivi un programma Python che consenta la registrazione delle iscrizioni di studenti ad una scuola di musica. 
Realizzare un menù con le seguenti funzioni: 
1) Aggiungi pagamento 
2) Mostra tutti i pagamenti 3) 
Ricerca pagamento per cognome 
4) Calcola il totale incassato"""



pagamenti = []  #lista vuota per i pagamenti
while True:
    print("Menu:")
    print("1) Aggiungi pagamento")
    print("2) Mostra tutti i pagamenti")
    print("3) Ricerca pagamento per cognome")
    print("4) Calcola il totale incassato")
    print("5) Esci")
    scelta = input("Scegli un'opzione (1-5): ")
    
    if scelta == '1':
        print("Inserisci il nome dello studente:")
        nome = input()
        print("Inserisci il cognome dello studente:")
        cognome = input()
        print("Inserisci l'importo del pagamento:")
        importo = float(input())
        pagamenti.append((nome, cognome, importo))  #aggiungo una tupla alla lista
        print("Pagamento aggiunto.")
    
    elif scelta == '2':
        print("Tutti i pagamenti:")
        for pagamento in pagamenti:
            print("Studente:", pagamento[0], pagamento[1], "Importo:", pagamento[2])
    
    elif scelta == '3':
        print("Inserisci il cognome da cercare:")
        cognome_ricerca = input()
        trovati = False
        for pagamento in pagamenti:
            if pagamento[1] == cognome_ricerca:
                print("Studente:", pagamento[0], pagamento[1], "Importo:", pagamento[2])
                trovati = True
        if not trovati:
            print("Nessun pagamento trovato per il cognome:", cognome_ricerca)
    
    elif scelta == '4':
        totale = sum(pagamento[2] for pagamento in pagamenti)
        print("Totale incassato:", totale)
    
    elif scelta == '5':
        print("Uscita dal programma.")
        break
    
    else:
        print("Opzione non valida. Riprova.")