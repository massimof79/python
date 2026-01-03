# Esercizio di Regressione Lineare con Scikit-Learn

## Titolo: Previsione dei Consumi Energetici di un Edificio Scolastico

### Contesto Reale

L'istituto tecnico "Fermi" vuole ottimizzare i consumi energetici della propria struttura per ridurre i costi e l'impatto ambientale. Il responsabile della manutenzione ha raccolto dati mensili relativi ai consumi di energia elettrica (in kWh) degli ultimi 2 anni, mettendoli in relazione con diversi fattori:

- **Temperatura media mensile** (°C)
- **Numero di giorni di lezione** nel mese
- **Numero di studenti presenti** (media mensile)
- **Ore di utilizzo laboratori informatici** (totale mensile)

### Obiettivo

Sviluppare un modello di **regressione lineare multipla** che permetta di:

1. Prevedere il consumo energetico mensile in base ai fattori sopra elencati
2. Identificare quali fattori influenzano maggiormente i consumi
3. Stimare il consumo previsto per i prossimi mesi per pianificare il budget energetico

### Compiti da Svolgere

#### 1. Analisi Esplorativa dei Dati
- Caricare il dataset `consumi_energetici.csv`
- Visualizzare le prime righe e le statistiche descrittive
- Creare grafici di correlazione tra le variabili

#### 2. Preparazione dei Dati
- Separare le features (variabili indipendenti) dalla target (consumo energetico)
- Dividere il dataset in training set (80%) e test set (20%)
- Applicare la normalizzazione delle features se necessario

#### 3. Addestramento del Modello
- Creare e addestrare un modello di regressione lineare
- Visualizzare i coefficienti del modello per interpretare l'importanza delle variabili

#### 4. Valutazione delle Prestazioni
- Calcolare le metriche di valutazione:
  - R² (coefficiente di determinazione)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
- Confrontare le previsioni con i valori reali attraverso grafici

#### 5. Previsioni Future
- Utilizzare il modello per prevedere i consumi dei prossimi 3 mesi basandosi su dati stimati

### Risultati Attesi

Al termine dell'esercizio, gli studenti dovranno essere in grado di:

- Comprendere come la regressione lineare modelli relazioni tra variabili
- Interpretare i coefficienti del modello per capire l'impatto di ciascuna variabile
- Valutare la qualità delle previsioni attraverso metriche appropriate
- Applicare il modello per fare previsioni su nuovi dati

### Domande di Riflessione

1. Quale fattore influenza maggiormente i consumi energetici? Perché?
2. Il modello è sufficientemente accurato per essere utilizzato nella pianificazione del budget?
3. Quali potrebbero essere i limiti di questo modello?
4. Come potremmo migliorare le previsioni?

### Estensioni Possibili

- Confrontare la regressione lineare con altri algoritmi (Ridge, Lasso)
- Aggiungere features derivate (es: stagionalità, festività)
- Implementare la cross-validation per una valutazione più robusta
- Creare un'interfaccia grafica per inserire nuovi dati e ottenere previsioni
