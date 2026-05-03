""" I Pesi ($W$) e il Bias ($b$): Spiega che la rete ha trovato dei valori 
numerici per cui il voto del test d'ingresso magari "pesa" di più della maturità. Puoi visualizzarli nel codice '
'con print(rete_neurale.coefs_).La Sigmoide in azione: Nota come l'output non è solo "Sì/No", ma una probabilità 
(grazie a predict_proba). Se la sicurezza è vicina al 50%, la rete è "confusa".L'importanza dei dati: Cosa succede se '
'addestriamo la rete solo con studenti promossi? La rete diventerà "troppo ottimista" (Bias cognitivo). '
'Questo serve a introdurre il tema dell'etica dell'IA.Esperimento: Chiedi agli studenti di modificare i '
'dati nell'array X o di aggiungere un terzo parametro (es. "Ore di studio mensili") e vedere come cambia 
il comportamento della rete.Perché questo è un compito di realtà?Perché simula esattamente come le aziende 
(banche per i mutui, HR per i curricula) utilizzano gli algoritmi per automatizzare decisioni basate su dati storici, 
rendendo tangibile il concetto di funzione di attivazione e forward propagation che hanno studiato in teoria. """



import numpy as np
from sklearn.neural_network import MLPClassifier

# 1. DATI DI REALTÀ (Training Set)
# [Voto Maturità (60-100), Punteggio Test (0-100)]
X = np.array([
    [95, 85], [65, 40], [80, 70], [60, 55], 
    [100, 90], [70, 30], [85, 60], [62, 45]
])
y = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# 2. CREAZIONE E ADDESTRAMENTO
# Usiamo un solo neurone nascosto per rendere i pesi facili da leggere
rete_neurale = MLPClassifier(hidden_layer_sizes=(1,), 
                            activation='logistic', 
                            solver='lbfgs', 
                            max_iter=2000,
                            random_state=1) # Risultati riproducibili

print("--- FASE 1: Addestramento ---")
rete_neurale.fit(X, y)
print(f"Iterazioni effettuate: {rete_neurale.n_iter_}")
print(f"Precisione finale sul set di addestramento: {rete_neurale.score(X, y) * 100}%\n")

# 3. ISPEZIONE DEI PESI (Il "Cervello" della rete)
# Questi sono i W e i b della formula studiata!
pesi = rete_neurale.coefs_[0]
bias = rete_neurale.intercepts_[0]

print("--- FASE 2: Cosa ha imparato la rete? ---")
print(f"Peso per 'Voto Maturità': {pesi[0][0]:.4f}")
print(f"Peso per 'Punteggio Test': {pesi[1][0]:.4f}")
print(f"Bias (Soglia di attivazione): {bias[0]:.4f}")
print("\nNota: Il peso maggiore indica quale caratteristica è più importante per la rete.\n")

# 4. PREVISIONE E PROBABILITÀ
nuovi_studenti = np.array([[88, 75], [68, 35]])
previsioni = rete_neurale.predict(nuovi_studenti)
probabilita = rete_neurale.predict_proba(nuovi_studenti)

print("--- FASE 3: Test su nuovi dati ---")
for i, studente in enumerate(nuovi_studenti):
    p_ammissione = probabilita[i][1] * 100
    esito = "Ammesso" if previsioni[i] == 1 else "Non Ammesso"
    
    print(f"Studente {i+1} {studente}:")
    print(f"  -> Probabilità calcolata dalla Sigmoide: {p_ammissione:.2f}%")
    print(f"  -> Verdetto finale: {esito}\n")