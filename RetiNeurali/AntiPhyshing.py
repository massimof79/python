"""
ESERCIZIO: FILTRO ANTI-PHISHING INTELLIGENTE
-------------------------------------------
OBIETTIVO: 
Addestrare una rete neurale a riconoscere email truffaldine (Phishing).
L'IA deve imparare a bilanciare due fattori di rischio che, presi singolarmente,
potrebbero non bastare a emettere una condanna.

CONCETTI CHIAVE DA OSSERVARE:
1. IL CALCOLO DEL LIVELLO: La rete moltiplica il numero di link e di errori 
   per i pesi (W) che ha imparato, sommando il Bias (b).
2. BACKPROPAGATION: Inizialmente la rete sbaglierà molti giudizi. 
   L'algoritmo tornerà indietro per correggere i pesi finché non riconoscerà le truffe.
3. COMPITO DI REALTÀ: Un'email con 10 link potrebbe essere una newsletter lecita,
   mentre un'email con 10 link E 5 errori grammaticali è quasi certamente phishing.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

# 1. DATI DI ADDESTRAMENTO (Esperienze passate)
# Input: [N. Link nell'email, N. Errori Grammaticali]
X = np.array([
    [1, 0], [2, 1],   # Mail normali (pochi link, zero errori)
    [15, 8], [12, 5], # Phishing evidente (molti link e molti errori)
    [8, 1], [10, 0],  # Newsletter (molti link, ma scritte bene) -> SICURE
    [2, 6], [1, 7]    # Mail scritte male ma senza link sospetti -> SICURE
])

# Output: 1 = PERICOLO (Phishing), 0 = SICURA (Ham)
y = np.array([0, 0, 1, 1, 0, 0, 0, 0])

# 2. CREAZIONE DELLA RETE NEURALE
# - Usiamo 'tanh' come funzione di attivazione: è centrata sullo zero (-1, 1) 
#   e aiuta a distinguere meglio le classi quando i dati sono contrastanti.
# - Un solo livello nascosto da 3 neuroni (struttura semplice).
filtro_antispam = MLPClassifier(hidden_layer_sizes=(3,), 
                                activation='tanh', 
                                solver='lbfgs', 
                                max_iter=1000,
                                random_state=42)

print("--- ANALISI TRAFFICO EMAIL ---")
print("L'IA sta imparando a distinguere una newsletter da una truffa...")
filtro_antispam.fit(X, y)
print(f"Modello addestrato con successo.\n")

# 3. SPIEGAZIONE MATEMATICA (Ispezione dei Pesi)
pesi_input = filtro_antispam.coefs_[0]
print("--- PESI DELLE CARATTERISTICHE (W) ---")
print(f"Pesi per 'Numero Link':\n{pesi_input[0]}")
print(f"Pesi per 'Errori Grammaticali':\n{pesi_input[1]}")
print("\nNota: I pesi sono distribuiti sui 3 neuroni del livello nascosto.")
print("Ognuno di essi sta cercando una diversa combinazione di sospetto.\n")

# 4. TEST SUL CAMPO (Compito di realtà)
# Caso 1: Email con 12 link ma grammatica perfetta (Newsletter?)
# Caso 2: Email con 14 link e 6 errori (Truffa?)
email_sospette = np.array([[12, 0], [14, 6]])

decisioni = filtro_antispam.predict(email_sospette)
sicurezza = filtro_antispam.predict_proba(email_sospette)

print("--- CONTROLLO SICUREZZA IN CORSO ---")
for i, email in enumerate(email_sospette):
    verdetto = "PERICOLOSA (Bloccata)" if decisioni[i] == 1 else "SICURA (In arrivo)"
    prob_pericolo = sicurezza[i][1] * 100
    
    print(f"Email {i+1} [Link: {email[0]}, Errori: {email[1]}]:")
    print(f"  > Verdetto: {verdetto}")
    print(f"  > Livello di rischio calcolato: {prob_pericolo:.2f}%\n")