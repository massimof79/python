"""Scrivi un programma python che consenta di registrare le prenotazioni di camere in un hotel. 
Realizza un menù con le seguenti funzioni 
1) aggiungi prenotazione 
2) elimina prenotazione per numero di stanza 
3) visualizza tutte le prenotazioni 
4) cerca prenotazione per numero di stanza. 
Per ogni prenotazione registrare Cognome, Nome, Stanza, DataCheckIn, DataCheckOut, Prezzo"""

#Array di prenotazioni
prenotazioni = []
while True:
    print("Menù:")
    print("1: Inserisci prenotazione: ")
    print("2: Elimina prenotazione di una stanza")
    print("3: Visualizza tutte le prenotazioni")
    print("4: Cerca prenotazione per numero di stanza")
    scelta = input("Scegli un'opzione tra 1 e 4")

    if scelta=='1':
        print("Inserisci il numero di stanza")
        stanza = input()
        print("Inserisci il Cognome dell'ospite")
        cognome = input()
        print("Inserisci il nome dell'ospite")
        nome = input()
        print("Inserisci il prezzo della stanza")
        prezzo = input()
        print("Inserisci la data di check-in (es: 2024-06-01)")
        data_checkin = input()
        print("Inserisci la data di check-out (es: 2024-06-05)")
        data_checkout = input()
        prenotazioni.append((cognome, nome, stanza, data_checkin, data_checkout, prezzo))

    elif scelta == '2':
        print("Inserisci il numero della stanza da eliminare")
        stanza = input()
        for stanza in prenotazioni:
            if stanza[1] == stanza:
               prenotazioni.remove()

    elif scelta=='3':
        for stanza in prenotazioni:
            print(stanza)


    elif scelta=='4':
        stanza = input()
        for stanza in prenotazioni:
            if stanza[1] == stanza:
                print(stanza)
           



    
    