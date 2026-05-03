"""
ESERCIZIO: L'IRRIGATORE AUTONOMO (Smart Farming)
------------------------------------------------
OBIETTIVO: 
Creare una rete neurale che decida se attivare l'irrigazione basandosi su 
Umidità (%) e Temperatura (°C). L'IA deve imparare una regola non lineare: 
"Irriga se è secco, ma non se fa troppo caldo (per evitare l'evaporazione)".

CONCETTI CHIAVE DA OSSERVARE:
1. INPUT (a^0): I dati grezzi dei sensori (Umidità, Temperatura).
2. LIVELLI NASCOSTI: Usiamo la funzione ReLU per elaborare i dati. 
   Ogni neurone cerca di capire una 'caratteristica' (es. "C'è rischio siccità?").
3. OUTPUT (a^L): Usiamo la Sigmoide per ottenere un valore tra 0 e 1,
   ovvero la probabilità che l'irrigazione debba essere accesa.
4. PESI (W) e BIAS (b): Noterete come la rete assegna importanza diversa 
   ai sensori durante l'addestramento tramite la Backpropagation.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

# 1. DATI DEI SENSORI (Training Set - L'esperienza passata)
# Input: [Umidità %, Temperatura °C]
X = np.array([
    [10, 20], [15, 25], # CASI A: Terreno secco, clima mite -> IRRIGARE (1)
    [80, 22], [90, 18], # CASI B: Terreno già molto umido -> NON IRRIGARE (0)
    [20, 42], [10, 45], # CASI C: Secco ma CALDO ESTREMO -> NON IRRIGARE (0) - Efficienza idrica
    [40, 25], [35, 28]  # CASI D: Situazione intermedia -> IRRIGARE (1)
])

# Output: 1 = Irrigazione ON (Sì), 0 = Irrigazione OFF (No)
y = np.array([1, 1, 0, 0, 0, 0, 1, 1])

# 2. CONFIGURAZIONE DELLA RETE NEURALE
# - hidden_layer_sizes=(4, 2): Due livelli intermedi. Il primo da 4 neuroni, il secondo da 2.
# - activation='relu': Funzione di attivazione per i calcoli interni (veloce ed efficiente).
# - max_iter=5000: Quante "ripetizioni" (epoca) fa la backpropagation per correggere i pesi.
rete_irrigazione = MLPClassifier(hidden_layer_sizes=(4, 2), 
                                 activation='relu', 
                                 solver='adam', 
                                 max_iter=5000,
                                 random_state=1)

print("--- FASE DI APPRENDIMENTO ---")
print("La rete sta analizzando i dati storici e calcolando le derivate...")
rete_irrigazione.fit(X, y)
print(f"Apprendimento completato in {rete_irrigazione.n_iter_} iterazioni.\n")

# 3. ISPEZIONE DEI PARAMETRI (W e b)
print("--- COSA HA IMPARATO L'IA? (Parametri Interni) ---")
# coeff_ contiene le matrici W dei vari livelli
for i, matrice_W in enumerate(rete_irrigazione.coefs_):
    print(f"Matrice dei pesi W del Livello {i} (dimensioni): {matrice_W.shape}")

# L'ultimo bias ci dice la "propensione" finale ad attivarsi
print(f"Bias finale del livello di output: {rete_irrigazione.intercepts_[-1][0]:.4f}\n")

# 4. COMPITO DI REALTÀ (Test su situazioni ignote)
# Situazione 1: Terreno al 25% (secco) e 30° (caldo ma accettabile)
# Situazione 2: Terreno al 15% (molto secco) ma 43° (bollente!)
nuovi_dati = np.array([[25, 30], [15, 43]]) 

previsioni = rete_irrigazione.predict(nuovi_dati)
probabilita = rete_irrigazione.predict_proba(nuovi_dati)

print("--- VERDETTO AUTOMATIZZATO ---")
for i, sensori in enumerate(nuovi_dati):
    esito = "ATTIVARE" if previsioni[i] == 1 else "NON ATTIVARE"
    # La probabilità viene dalla funzione Sigmoide applicata alla fine
    confidenza = probabilita[i][previsioni[i]] * 100
    
    print(f"Sensori [Umidità={sensori[0]}%, Temp={sensori[1]}°C]:")
    print(f"  > Decisione: {esito}")
    print(f"  > Sicurezza del calcolo: {confidenza:.2f}%\n")