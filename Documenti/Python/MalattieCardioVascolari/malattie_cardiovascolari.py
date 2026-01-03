"""
Sistema di Predizione del Rischio Cardiovascolare
Programma educativo per studenti - Classificazione binaria
Versione semplificata con 2 modelli
"""

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================================================================
# PARTE 1: CARICAMENTO E PREPARAZIONE DEI DATI
# =============================================================================

def carica_dataset():
    """
    Carica il dataset delle malattie cardiache da UCI
    """
    print(" Caricamento dataset...")
    
    # URL del dataset pubblico UCI Heart Disease
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Nomi delle colonne del dataset
    colonne = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
               'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
               'ca', 'thal', 'target']
    
    # Carica i dati nel dataset
    df = pd.read_csv(url, names=colonne, na_values='?')
    
    # Rimuovi righe con valori mancanti
    df = df.dropna()
    
    # Converti target in binario (0 = sano, 1 = malattia)
    # Nel dataset originale: 0=no malattia, 1-4=presenza malattia
    df['target'] = (df['target'] > 0).astype(int)
    

    print(df)

    print(f" Dataset caricato: {len(df)} pazienti")
    print(f"   - Pazienti sani: {(df['target']==0).sum()}")
    print(f"   - Pazienti a rischio: {(df['target']==1).sum()}")
    
    return df

def mostra_statistiche(df):
    """
    Mostra statistiche descrittive del dataset
    """
    print("\n STATISTICHE DEL DATASET")
    print("="*60)
    print(df.describe())
    print("\n Distribuzione per sesso:")
    print(df['sex'].value_counts())
    print("   (1 = Maschio, 0 = Femmina)")

# =============================================================================
# PARTE 2: ADDESTRAMENTO DEL MODELLO
# =============================================================================

def prepara_dati(df):
    """
    Prepara i dati per l'addestramento
    """
    print("\n Preparazione dati per l'addestramento...")
    
    # Separa features (X) e target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Dividi in training set (80%) e test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizza i dati (importante per la Regressione Logistica)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f" Dati preparati:")
    print(f"   - Training set: {len(X_train)} pazienti")
    print(f"   - Test set: {len(X_test)} pazienti")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns

def addestra_modelli(X_train, X_test, y_train, y_test):
    """
    Addestra e confronta Regressione Logistica e Albero Decisionale
    """
    print("\n ADDESTRAMENTO DEI MODELLI")
    print("="*60)
    
    # Definisci i due modelli
    modelli = {
        'Regressione Logistica': LogisticRegression(max_iter=1000, random_state=42),
        'Albero Decisionale': DecisionTreeClassifier(max_depth=5, random_state=42)
    }
    
    risultati = {}
    
    for nome, modello in modelli.items():
        print(f"\n Addestramento: {nome}...")
        
        # Descrizione del modello
        if nome == 'Regressione Logistica':
            print("    Modello statistico che calcola probabilità")
            print("      Vantaggi: veloce, interpretabile, probabilità calibrate")
        else:
            print("    Modello basato su regole decisionali (if-then)")
            print("      Vantaggi: intuitivo, gestisce relazioni non-lineari")
        
        # Addestra il modello
        modello.fit(X_train, y_train)
        
        # Predizioni
        y_pred_train = modello.predict(X_train)
        y_pred_test = modello.predict(X_test)
        
        # Calcola accuratezza
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        
        risultati[nome] = {
            'modello': modello,
            'acc_train': acc_train,
            'acc_test': acc_test,
            'y_pred': y_pred_test
        }
        
        print(f"    Accuratezza Training: {acc_train:.2%}")
        print(f"    Accuratezza Test: {acc_test:.2%}")
    
    # Confronta i modelli
    print("\n CONFRONTO TRA I MODELLI")
    print("="*60)
    for nome, dati in risultati.items():
        print(f"{nome}:")
        print(f"  - Accuratezza Test: {dati['acc_test']:.2%}")
        print(f"  - Overfitting: {abs(dati['acc_train'] - dati['acc_test']):.2%}")
    
    # Scegli il modello migliore
    modello_migliore = max(risultati.items(), key=lambda x: x[1]['acc_test'])
    
    print(f"\n MODELLO MIGLIORE: {modello_migliore[0]}")
    print(f"   Accuratezza Test: {modello_migliore[1]['acc_test']:.2%}")
    
    return risultati, modello_migliore

def mostra_valutazione(y_test, y_pred, nome_modello):
    """
    Mostra metriche di valutazione dettagliate
    """
    print(f"\n VALUTAZIONE DETTAGLIATA - {nome_modello}")
    print("="*60)
    
    # Report di classificazione
    print("\n Report di Classificazione:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Sano', 'A Rischio']))
    
    # Matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    print("\n Matrice di Confusione:")
    print("                  Predetto")
    print("                Sano  A Rischio")
    print(f"Reale Sano      {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"      A Rischio {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Spiegazione della matrice
    print("\n Spiegazione:")
    print(f"   - Veri Negativi (VN): {cm[0][0]} pazienti sani identificati correttamente")
    print(f"   - Falsi Positivi (FP): {cm[0][1]} pazienti sani classificati erroneamente a rischio")
    print(f"   - Falsi Negativi (FN): {cm[1][0]} pazienti a rischio non identificati ⚠️")
    print(f"   - Veri Positivi (VP): {cm[1][1]} pazienti a rischio identificati correttamente")

# =============================================================================
# PARTE 3: SISTEMA DI PREDIZIONE PER NUOVI PAZIENTI
# =============================================================================

def predici_rischio_paziente(modello, scaler, feature_names):
    """
    Sistema interattivo per predire il rischio di un nuovo paziente
    """
    print("\n" + "="*60)
    print(" SISTEMA DI PREDIZIONE RISCHIO CARDIOVASCOLARE")
    print("="*60)
    
    print("\nInserisci i parametri del paziente:\n")
    
    # Dizionario per memorizzare i valori
    paziente = {}
    
    # Richiedi input per ogni parametro
    domande = {
        'age': "Età (anni): ",
        'sex': "Sesso (1=Maschio, 0=Femmina): ",
        'cp': "Tipo dolore toracico (0=tipica, 1=atipica, 2=non-anginosa, 3=asintomatico): ",
        'trestbps': "Pressione arteriosa a riposo (mmHg): ",
        'chol': "Colesterolo totale (mg/dl): ",
        'fbs': "Glicemia a digiuno >120 mg/dl (1=Sì, 0=No): ",
        'restecg': "ECG a riposo (0=normale, 1=anomalie ST-T, 2=ipertrofia): ",
        'thalach': "Frequenza cardiaca massima: ",
        'exang': "Angina da esercizio (1=Sì, 0=No): ",
        'oldpeak': "Depressione ST (es. 2.3): ",
        'slope': "Pendenza ST (0-2): ",
        'ca': "Numero vasi colorati (0-3): ",
        'thal': "Talassemia (1=normale, 2=difetto fisso, 3=difetto reversibile): "
    }
    
    try:
        for feature in feature_names:
            valore = float(input(domande[feature]))
            paziente[feature] = valore
        
        # Crea array per la predizione
        dati_paziente = np.array([list(paziente.values())])
        
        # Normalizza i dati
        dati_normalizzati = scaler.transform(dati_paziente)
        
        # Predici
        predizione = modello.predict(dati_normalizzati)[0]
        probabilita = modello.predict_proba(dati_normalizzati)[0]
        
        # Mostra risultato
        print("\n" + "="*60)
        print(" RISULTATO DELLA PREDIZIONE")
        print("="*60)
        
        if predizione == 0:
            print(" PAZIENTE A BASSO RISCHIO")
            print(f"   Probabilità di essere sano: {probabilita[0]:.1%}")
            print(f"   Probabilità di rischio: {probabilita[1]:.1%}")
        else:
            print("   PAZIENTE A RISCHIO")
            print(f"   Probabilità di malattia cardiovascolare: {probabilita[1]:.1%}")
            print(f"   Probabilità di essere sano: {probabilita[0]:.1%}")
        
        print("\n💡 Nota: Questa è una predizione automatica.")
        print("   Consultare sempre un medico per una diagnosi definitiva.")
        
    except ValueError:
        print("\n Errore: Inserire valori numerici validi!")
    except Exception as e:
        print(f"\n  Errore: {e}")

def esempio_predizione(modello, scaler, feature_names):
    """
    Esempio di predizione con dati di test
    """
    print("\n" + "="*60)
    print(" ESEMPI DI PREDIZIONE")
    print("="*60)
    
    # Paziente esempio a basso rischio
    paziente_sano = {
        'age': 45, 'sex': 1, 'cp': 3, 'trestbps': 120, 'chol': 200,
        'fbs': 0, 'restecg': 0, 'thalach': 170, 'exang': 0,
        'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 2
    }
    
    # Paziente esempio a rischio
    paziente_rischio = {
        'age': 65, 'sex': 1, 'cp': 0, 'trestbps': 160, 'chol': 280,
        'fbs': 1, 'restecg': 2, 'thalach': 120, 'exang': 1,
        'oldpeak': 3.5, 'slope': 2, 'ca': 2, 'thal': 3
    }
    
    esempi = [
        ("🟢 Esempio 1: Paziente Sano", paziente_sano),
        ("🔴 Esempio 2: Paziente a Rischio", paziente_rischio)
    ]
    
    for nome, paz in esempi:
        dati = np.array([list(paz.values())])
        dati_norm = scaler.transform(dati)
        pred = modello.predict(dati_norm)[0]
        prob = modello.predict_proba(dati_norm)[0]
        
        print(f"\n{nome}:")
        print(f"  Parametri chiave: Età={paz['age']}, Colesterolo={paz['chol']}, "
              f"Pressione={paz['trestbps']}")
        print(f"  Predizione: {' Sano' if pred == 0 else '  A Rischio'}")
        print(f"  Probabilità rischio: {prob[1]:.1%}")

# =============================================================================
# PROGRAMMA PRINCIPALE
# =============================================================================

def main():
    """
    Funzione principale del programma
    """
    print("\n" + "="*60)
    print("🏥 SISTEMA DI PREDIZIONE MALATTIE CARDIOVASCOLARI")
    print("   Versione con 2 modelli: Regressione Logistica e Albero Decisionale")
    print("="*60)
    
    # 1. Carica dataset
    df = carica_dataset()
    
    mostra_statistiche(df)

    sys.exit()

    # 2. Prepara i dati
    X_train, X_test, y_train, y_test, scaler, feature_names = prepara_dati(df)
    
    # 3. Addestra i modelli
    risultati, (nome_migliore, dati_migliore) = addestra_modelli(
        X_train, X_test, y_train, y_test
    )
    
    # 4. Mostra valutazione dettagliata del modello migliore
    mostra_valutazione(y_test, dati_migliore['y_pred'], nome_migliore)
    
    # 5. Modello finale
    modello_finale = dati_migliore['modello']
    
    # 6. Esempi di predizione
    esempio_predizione(modello_finale, scaler, feature_names)
    
    # 7. Menu interattivo
    while True:
        print("\n" + "="*60)
        print(" MENU OPZIONI")
        print("="*60)
        print("1. Predire rischio per un nuovo paziente")
        print("2. Mostrare confronto tra i modelli")
        print("3. Uscire")
        
        scelta = input("\nScegli un'opzione (1-3): ")
        
        if scelta == '1':
            predici_rischio_paziente(modello_finale, scaler, feature_names)
        elif scelta == '2':
            print("\n CONFRONTO MODELLI:")
            for nome, dati in risultati.items():
                mostra_valutazione(y_test, dati['y_pred'], nome)
        elif scelta == '3':
            print("\n Programma terminato.")
            print("="*60)
            break
        else:
            print("Opzione non valida!") 

# Esegui il programma
if __name__ == "__main__":
    main()