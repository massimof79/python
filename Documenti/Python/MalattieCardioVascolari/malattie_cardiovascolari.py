"""
Sistema di Predizione del Rischio Cardiovascolare
Programma educativo per studenti - Versione semplificata
Usa solo Regressione Logistica
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================================================================
# PARTE 1: CARICAMENTO DEI DATI
# =============================================================================

def carica_dataset():
    """
    Scarica il dataset delle malattie cardiache da internet
    Ritorna: DataFrame con i dati dei pazienti
    """
    print("Caricamento dataset...")
    
    # URL del dataset pubblico
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Nomi delle 14 colonne del dataset
    colonne = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
               'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
               'ca', 'thal', 'target']
    
    # Scarica i dati (il punto interrogativo '?' indica valori mancanti)
    df = pd.read_csv(url, names=colonne, na_values='?')
    
    # Elimina le righe con dati mancanti
    df = df.dropna()
    
    # Trasforma target in binario: 0 = sano, 1 = malattia
    # (nel dataset originale: 0=sano, 1-4=vari livelli di malattia)
    df['target'] = (df['target'] > 0).astype(int)
    
    # Mostra informazioni sul dataset
    print(f"Dataset caricato: {len(df)} pazienti")
    print(f"   - Pazienti sani: {(df['target']==0).sum()}")
    print(f"   - Pazienti a rischio: {(df['target']==1).sum()}")
    
    return df

# =============================================================================
# PARTE 2: PREPARAZIONE E ADDESTRAMENTO
# =============================================================================

def prepara_e_addestra(df):
    """
    Prepara i dati e addestra il modello di Regressione Logistica
    Ritorna: modello addestrato, scaler, nomi delle features e dati di test
    """
    print("\nPreparazione dati...")
    
    # STEP 1: Separa le features (X) dal target (y)
    # X = tutte le colonne tranne 'target'
    # y = solo la colonna 'target' (0 o 1)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # STEP 2: Dividi in training set (80%) e test set (20%)
    # training set = per addestrare il modello
    # test set = per valutare le prestazioni
    # stratify=y mantiene le stesse proporzioni di sani/malati
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   - Training set: {len(X_train)} pazienti")
    print(f"   - Test set: {len(X_test)} pazienti")
    
    # STEP 3: Normalizza i dati (porta tutti i valori su scala simile)
    # Importante: prima calcola media e deviazione standard sul training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Poi applica la stessa trasformazione al test set
    X_test_scaled = scaler.transform(X_test)
    
    print("\nAddestramento del modello...")
    print("   Modello: Regressione Logistica")
    print("   - Calcola la probabilita' che un paziente sia a rischio")
    print("   - Veloce e interpretabile")
    
    # STEP 4: Crea e addestra il modello
    # max_iter=1000 = numero massimo di iterazioni per l'ottimizzazione
    # random_state=42 = per risultati riproducibili
    modello = LogisticRegression(max_iter=1000, random_state=42)
    modello.fit(X_train_scaled, y_train)
    
    # STEP 5: Valuta il modello
    # Predici sul training set e test set
    y_pred_train = modello.predict(X_train_scaled)
    y_pred_test = modello.predict(X_test_scaled)
    
    # Calcola l'accuratezza (% di predizioni corrette)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print(f"\nRisultati:")
    print(f"   - Accuratezza Training: {acc_train:.2%}")
    print(f"   - Accuratezza Test: {acc_test:.2%}")
    
    # Se l'accuratezza sul training è molto più alta del test = overfitting
    diff = abs(acc_train - acc_test)
    if diff > 0.10:
        print(f"   ATTENZIONE: Possibile overfitting: differenza {diff:.2%}")
    
    return modello, scaler, X.columns, X_test_scaled, y_test

# =============================================================================
# PARTE 3: VALUTAZIONE DETTAGLIATA
# =============================================================================

def mostra_valutazione(modello, X_test, y_test):
    """
    Mostra metriche di valutazione dettagliate del modello
    """
    print("\n" + "="*60)
    print("VALUTAZIONE DETTAGLIATA DEL MODELLO")
    print("="*60)
    
    # Predici sul test set
    y_pred = modello.predict(X_test)
    
    # Report di classificazione (precision, recall, f1-score)
    print("\nReport di Classificazione:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Sano', 'A Rischio']))
    
    # Matrice di confusione
    # Mostra quante predizioni corrette/sbagliate per ogni classe
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nMatrice di Confusione:")
    print("                  Predetto")
    print("                Sano  A Rischio")
    print(f"Reale Sano      {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"      A Rischio {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    print("\nInterpretazione:")
    print(f"   - Veri Negativi: {cm[0][0]} pazienti sani identificati correttamente")
    print(f"   - Falsi Positivi: {cm[0][1]} pazienti sani classificati erroneamente a rischio")
    print(f"   - Falsi Negativi: {cm[1][0]} pazienti a rischio NON identificati (PERICOLOSO!)")
    print(f"   - Veri Positivi: {cm[1][1]} pazienti a rischio identificati correttamente")

# =============================================================================
# PARTE 4: PREDIZIONE PER NUOVI PAZIENTI
# =============================================================================

def esempio_predizioni(modello, scaler, feature_names):
    """
    Mostra esempi di predizione con casi tipici
    """
    print("\n" + "="*60)
    print("ESEMPI DI PREDIZIONE")
    print("="*60)
    
    # Esempio 1: Paziente sano (giovane, parametri normali)
    paziente_sano = {
        'age': 45, 'sex': 1, 'cp': 3, 'trestbps': 120, 'chol': 200,
        'fbs': 0, 'restecg': 0, 'thalach': 170, 'exang': 0,
        'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 2
    }
    
    # Esempio 2: Paziente a rischio (anziano, parametri alterati)
    paziente_rischio = {
        'age': 65, 'sex': 1, 'cp': 0, 'trestbps': 160, 'chol': 280,
        'fbs': 1, 'restecg': 2, 'thalach': 120, 'exang': 1,
        'oldpeak': 3.5, 'slope': 2, 'ca': 2, 'thal': 3
    }
    
    # Analizza entrambi i pazienti
    esempi = [
        ("Esempio 1: Paziente con profilo sano", paziente_sano),
        ("Esempio 2: Paziente con profilo a rischio", paziente_rischio)
    ]
    
    for nome, paziente in esempi:
        # Converti il dizionario in DataFrame per mantenere i nomi delle colonne
        dati_df = pd.DataFrame([paziente], columns=feature_names)
        # Normalizza i dati (come fatto nel training)
        dati_norm = scaler.transform(dati_df)
        # Predici classe e probabilità
        predizione = modello.predict(dati_norm)[0]
        probabilita = modello.predict_proba(dati_norm)[0]
        
        print(f"\n{nome}:")
        print(f"  Eta': {paziente['age']} anni, Colesterolo: {paziente['chol']} mg/dl, "
              f"Pressione: {paziente['trestbps']} mmHg")
        print(f"  Predizione: {'SANO' if predizione == 0 else 'A RISCHIO'}")
        print(f"  Probabilita' di rischio: {probabilita[1]:.1%}")

def predici_nuovo_paziente(modello, scaler, feature_names):
    """
    Permette di inserire i dati di un nuovo paziente e ottenere una predizione
    """
    print("\n" + "="*60)
    print("PREDIZIONE PER NUOVO PAZIENTE")
    print("="*60)
    
    # Descrizione delle features da inserire
    descrizioni = {
        'age': "Eta' del paziente (anni): ",
        'sex': "Sesso (1=Maschio, 0=Femmina): ",
        'cp': "Tipo dolore toracico (0-3, 3=asintomatico): ",
        'trestbps': "Pressione arteriosa a riposo (mmHg, es. 120): ",
        'chol': "Colesterolo totale (mg/dl, es. 200): ",
        'fbs': "Glicemia a digiuno >120 mg/dl (1=Si', 0=No): ",
        'restecg': "ECG a riposo (0=normale, 1-2=anomalie): ",
        'thalach': "Frequenza cardiaca massima (es. 150): ",
        'exang': "Angina da esercizio (1=Si', 0=No): ",
        'oldpeak': "Depressione ST (numero decimale, es. 2.3): ",
        'slope': "Pendenza ST (0-2): ",
        'ca': "Numero vasi colorati in fluoroscopia (0-3): ",
        'thal': "Talassemia (1=normale, 2=difetto fisso, 3=reversibile): "
    }
    
    print("\nInserisci i parametri del paziente:\n")
    
    try:
        # Raccogli i dati del paziente
        paziente = {}
        for feature in feature_names:
            valore = float(input(descrizioni[feature]))
            paziente[feature] = valore
        
        # Prepara i dati per la predizione usando DataFrame
        # In questo modo manteniamo i nomi delle colonne ed evitiamo il warning
        dati_paziente_df = pd.DataFrame([paziente], columns=feature_names)
        dati_normalizzati = scaler.transform(dati_paziente_df)
        
        # Esegui la predizione
        predizione = modello.predict(dati_normalizzati)[0]
        probabilita = modello.predict_proba(dati_normalizzati)[0]
        
        # Mostra il risultato
        print("\n" + "="*60)
        print("RISULTATO DELLA PREDIZIONE")
        print("="*60)
        
        if predizione == 0:
            print("\nPAZIENTE A BASSO RISCHIO")
            print(f"   Probabilita' di essere sano: {probabilita[0]:.1%}")
            print(f"   Probabilita' di rischio cardiovascolare: {probabilita[1]:.1%}")
        else:
            print("\nPAZIENTE A RISCHIO")
            print(f"   Probabilita' di malattia cardiovascolare: {probabilita[1]:.1%}")
            print(f"   Probabilita' di essere sano: {probabilita[0]:.1%}")
        
        print("\nIMPORTANTE: Questa e' solo una predizione automatica.")
        print("   Consultare sempre un medico per una diagnosi definitiva!")
        
    except ValueError:
        print("\nERRORE: Inserire solo valori numerici validi!")
    except Exception as e:
        print(f"\nERRORE imprevisto: {e}")

# =============================================================================
# PROGRAMMA PRINCIPALE
# =============================================================================

def main():
    """
    Funzione principale che coordina tutto il programma
    """
    print("\n" + "="*70)
    print("SISTEMA DI PREDIZIONE RISCHIO MALATTIE CARDIOVASCOLARI")
    print("   Programma educativo con Regressione Logistica")
    print("="*70)
    
    # PASSO 1: Carica il dataset
    df = carica_dataset()
    
    # PASSO 2: Prepara i dati e addestra il modello
    modello, scaler, feature_names, X_test, y_test = prepara_e_addestra(df)
    
    # PASSO 3: Mostra valutazione dettagliata
    mostra_valutazione(modello, X_test, y_test)
    
    # PASSO 4: Mostra esempi di predizione
    esempio_predizioni(modello, scaler, feature_names)
    
    # PASSO 5: Menu interattivo
    while True:
        print("\n" + "="*60)
        print("MENU OPZIONI")
        print("="*60)
        print("1. Predire rischio per un nuovo paziente")
        print("2. Mostrare nuovamente la valutazione del modello")
        print("3. Uscire dal programma")
        
        scelta = input("\nScegli un'opzione (1-3): ")
        
        if scelta == '1':
            predici_nuovo_paziente(modello, scaler, feature_names)
        elif scelta == '2':
            mostra_valutazione(modello, X_test, y_test)
        elif scelta == '3':
            print("\nProgramma terminato. Arrivederci!")
            print("="*60)
            break
        else:
            print("\nERRORE: Opzione non valida! Scegliere 1, 2 o 3.")

# Punto di ingresso del programma
if __name__ == "__main__":
    main()