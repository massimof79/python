## Copilot / Agent quick guide — Esercizi Python (workspace root)

Questo repository contiene esercizi Python a livello didattico. Le istruzioni qui sotto servono a far sì che un agente AI modifichi e migliori il codice senza introdurre regressioni o cambiare intenzione didattica.

1) Big picture
- Il progetto attuale è composto principalmente da file singoli, es.: `04_Liste_Tuple.py`.
- Pattern osservato: i dati studente/voto sono rappresentati come una lista di tuple: `voti.append((nome, cognome, voto, materia))`.
- Il codice è un programma CLI interattivo che usa `input()` e `print()`; non ci sono test o dipendenze esterne.

2) Cosa sa fare l'agente subito
- Piccole refactorizzazioni non invasive (estrarre funzioni, migliorare nomi locali) mantenendo i messaggi in italiano.
- Conversione dell'eseguibile interattivo in funzioni testabili (es. separare parsing, calcolo medie e stampa).
- Aggiungere file di supporto non invasivi: `README.md`, `.gitignore` (es. venv, __pycache__), e test minimalisti sotto `tests/` dopo aver refactorizzato input/output.

3) Regole operative specifiche
- Non cambiare il linguaggio dei prompt (sono in italiano): preserva le stringhe passate a `input()` e i `print()` a meno che il task non richieda esplicitamente la localizzazione.
- Mantieni la struttura dati esistente (lista di tuple) se l'esercizio è didattico; proponi una migrazione a dict/NamedTuple solo se spieghi vantaggi e aggiorni tutte le parti che la utilizzano.
- Quando crei test, non alterare il comportamento dell'interfaccia utente: estrai la logica pura in funzioni che accettano parametri e restituiscono valori.

4) Come eseguire / flussi di sviluppo (scopribili dal repo)
- Esecuzione rapida (script interattivo):
```bash
python3 04_Liste_Tuple.py
```
- Per eseguire in modo non-interattivo (temporaneo), reindirizzare stdin con here-doc o file: `python3 04_Liste_Tuple.py < input.txt` — ma preferisci refactor per test automatici.

5) Esempi concreti dal codice
- Input/record: `voti.append((nome, cognome, voto, materia))` — ogni tupla contiene esattamente 4 campi.
- Accesso ai campi via indice: `voti[i][2]` (voto) e `voti[i][3]` (materia). Quando refattorizzi, sostituisci gli accessi indicizzati con nomi chiari o helper.

6) Piccole attività a basso rischio che l'agente può proporre e implementare
- Aggiungere `README.md` con istruzioni per lanciare lo script.
- Aggiungere `.gitignore` (es.: `__pycache__/`, `.venv/`).
- Refactor: estrarre funzioni pure per calcolo medie e aggregazioni, lasciando intatta la CLI.

7) Cosa non fare senza conferma umana
- Cambiare il linguaggio dei messaggi o semplificare l'interazione didattica (es.: rimuovere richieste esplicite di nome/cognome).
- Rimuovere l'interattività senza fornire una versione che conserva la stessa UX.

8) Dove guardare quando cerchi pattern
- `04_Liste_Tuple.py` — esempio primario per: modello dati, accesso ai campi tramite indice, uso di `input()`.
- Cartella `massimof79/python/` — controllare prima di applicare regole globali (potrebbero esserci altri esercizi con convenzioni simili).

Feedback richiesto
- Segnala se vuoi che generi anche un `README.md` e le istruzioni per caricare il progetto su GitHub (posso aggiungere comandi `git` passo-passo).
