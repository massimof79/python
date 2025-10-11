"""
===============================================================================
ESERCIZIO NUMPY - ANALISI DATI METEO E QUALITÀ DELL'ARIA
===============================================================================

CONTESTO:
Sei un data analyst che lavora per un'agenzia ambientale. Ti è stato chiesto
di analizzare i dati meteo e di qualità dell'aria raccolti in una settimana
da 5 stazioni di monitoraggio distribuite nella tua città.

I dati raccolti includono:
- Temperature (°C)
- Livelli di PM10 (particolato fine, μg/m³)
- Umidità relativa (%)

Il tuo compito è elaborare questi dati per fornire un report alle autorità
locali con statistiche significative e individuare eventuali criticità.

===============================================================================
"""

import numpy as np

"""
-------------------------------------------------------------------------------
PARTE 1: RACCOLTA E PREPARAZIONE DEI DATI
-------------------------------------------------------------------------------
Simula la raccolta dei dati dalle 5 stazioni di monitoraggio
"""

# TODO 1.1: Crea un array con le temperature medie giornaliere (°C) 
# registrate dalla stazione centrale
# Dati: Lunedì=15.5, Martedì=17.2, Mercoledì=16.8, Giovedì=14.3, Venerdì=18.1, 
#       Sabato=19.5, Domenica=18.7
temperature_settimana = None

print("Temperature settimanali:", temperature_settimana)

# TODO 1.2: Crea un array con i livelli di PM10 (μg/m³) rilevati ogni giorno
# La soglia di allarme è 50 μg/m³
# Dati: 35, 42, 55, 48, 38, 31, 40
livelli_pm10 = None

# TODO 1.3: Genera casualmente i valori di umidità (%) per 7 giorni
# Usa np.random.uniform() per generare valori tra 40 e 80
umidita = None

# TODO 1.4: Crea un array con i giorni della settimana usando np.arange()
# Da 1 (Lunedì) a 7 (Domenica)
giorni = None

# TODO 1.5: Inizializza un array di zeri per memorizzare le anomalie rilevate
# (0 = nessuna anomalia, 1 = anomalia rilevata)
anomalie = None

"""
-------------------------------------------------------------------------------
PARTE 2: ANALISI STATISTICA DI BASE
-------------------------------------------------------------------------------
Calcola le statistiche principali per il report
"""

# TODO 2.1: Calcola la temperatura media della settimana
temp_media = None
print(f"\nTemperatura media settimanale: {temp_media}°C")

# TODO 2.2: Trova la temperatura massima registrata
temp_massima = None
print(f"Temperatura massima: {temp_massima}°C")

# TODO 2.3: Trova la temperatura minima registrata
temp_minima = None
print(f"Temperatura minima: {temp_minima}°C")

# TODO 2.4: Calcola l'escursione termica (differenza tra max e min)
escursione_termica = None

# TODO 2.5: Calcola la media dei livelli di PM10
pm10_medio = None
print(f"PM10 medio settimanale: {pm10_medio:.1f} μg/m³")

# TODO 2.6: Calcola la deviazione standard del PM10
# (indica quanto variano i valori rispetto alla media)
pm10_deviazione = None

"""
-------------------------------------------------------------------------------
PARTE 3: CONVERSIONI E TRASFORMAZIONI
-------------------------------------------------------------------------------
Converti i dati in altre unità di misura per il report internazionale
"""

# TODO 3.1: Converti le temperature da Celsius a Fahrenheit
# Formula: F = C × 9/5 + 32
temperature_fahrenheit = None

# TODO 3.2: Normalizza i livelli di PM10 rispetto alla soglia di sicurezza (50 μg/m³)
# Valori > 1 indicano superamento della soglia
pm10_normalizzato = None

# TODO 3.3: Calcola l'indice di comfort considerando temperatura e umidità
# Formula semplificata: comfort = temperatura - (umidità / 10)
# Valori ottimali tra 12 e 18
indice_comfort = None

"""
-------------------------------------------------------------------------------
PARTE 4: IDENTIFICAZIONE CRITICITÀ
-------------------------------------------------------------------------------
Individua i giorni con condizioni critiche
"""

# TODO 4.1: Identifica i giorni in cui il PM10 ha superato la soglia di 50 μg/m³
# Crea un array booleano
giorni_pm10_alto = None
print(f"\nGiorni con PM10 oltre soglia: {giorni_pm10_alto}")

# TODO 4.2: Estrai solo i valori di PM10 che hanno superato la soglia
valori_critici_pm10 = None

# TODO 4.3: Conta quanti giorni hanno superato la soglia
num_giorni_critici = None
print(f"Numero giorni critici per PM10: {num_giorni_critici}")

# TODO 4.4: Identifica i giorni con temperature estreme (sotto 15°C o sopra 19°C)
giorni_temp_estrema = None

# TODO 4.5: Trova l'indice del giorno con il PM10 più alto
giorno_peggiore = None
print(f"Giorno con inquinamento peggiore: giorno {giorno_peggiore + 1}")

"""
-------------------------------------------------------------------------------
PARTE 5: ANALISI MULTI-STAZIONE
-------------------------------------------------------------------------------
Analizza i dati provenienti da tutte le 5 stazioni di monitoraggio
"""

# TODO 5.1: Crea una matrice 5x7 con i dati di temperatura di tutte le stazioni
# Ogni riga rappresenta una stazione, ogni colonna un giorno
# Usa i dati della stazione centrale e genera altri valori casuali simili
# Suggerimento: usa np.random.normal(temperature_settimana, 2, size=(5,7))
temperature_multi_stazione = None

# TODO 5.2: Calcola la temperatura media per ogni giorno (media tra le 5 stazioni)
# Suggerimento: usa axis=0 per fare la media per colonna
temp_media_giornaliera = None

# TODO 5.3: Calcola la temperatura media di ogni stazione nella settimana
# Suggerimento: usa axis=1 per fare la media per riga
temp_media_per_stazione = None

# TODO 5.4: Trova quale stazione ha registrato la temperatura più alta in assoluto
# Suggerimento: usa np.max() sulla matrice completa
temp_max_assoluta = None

# TODO 5.5: Identifica la stazione più calda (quella con media più alta)
# Suggerimento: usa np.argmax() sulle medie per stazione
stazione_piu_calda = None
print(f"Stazione più calda: Stazione {stazione_piu_calda + 1}")

"""
-------------------------------------------------------------------------------
PARTE 6: TREND E PREVISIONI
-------------------------------------------------------------------------------
Analizza i trend settimanali
"""

# TODO 6.1: Calcola la variazione giornaliera delle temperature
# (temperatura del giorno N+1 meno temperatura del giorno N)
# Suggerimento: usa lo slicing e la sottrazione tra array
variazioni_temp = None

# TODO 6.2: Determina se la temperatura è generalmente in aumento o diminuzione
# Calcola la media delle variazioni: se positiva → trend in aumento
trend_temperatura = None

# TODO 6.3: Crea un array con i valori di PM10 ordinati dal più basso al più alto
pm10_ordinato = None

# TODO 6.4: Calcola il percentile 75 del PM10 
# (valore sotto il quale cade il 75% delle misurazioni)
pm10_percentile75 = None

"""
-------------------------------------------------------------------------------
PARTE 7: CREAZIONE REPORT FINALE
-------------------------------------------------------------------------------
Prepara le statistiche riassuntive per il report
"""

# TODO 7.1: Crea una matrice riassuntiva 3x7 con temperatura, PM10 e umidità
# Righe: [temperatura, pm10, umidità]
# Colonne: giorni della settimana
dati_completi = None

# TODO 7.2: Calcola le statistiche principali per ogni parametro
# Media di ogni riga (axis=1)
medie_parametri = None

# TODO 7.3: Arrotonda tutte le medie a 1 decimale
medie_arrotondate = None

# TODO 7.4: Crea un array di classificazione della qualità dell'aria
# 0 = Buona (PM10 < 40), 1 = Moderata (40-50), 2 = Scarsa (> 50)
qualita_aria = None

"""
-------------------------------------------------------------------------------
PARTE 8: ESPORTAZIONE E SALVATAGGIO
-------------------------------------------------------------------------------
Salva i risultati per condividerli
"""

# TODO 8.1: Salva la matrice temperature_multi_stazione in un file CSV
# Suggerimento: np.savetxt('temperature_stazioni.csv', matrice, delimiter=',')

# TODO 8.2: Crea un array strutturato con un riassunto giornaliero
# Include: giorno, temperatura, PM10, qualità aria
# Suggerimento: usa np.column_stack() per unire gli array

"""
-------------------------------------------------------------------------------
DOMANDE DI RIFLESSIONE
-------------------------------------------------------------------------------

Dopo aver completato l'esercizio, rispondi a queste domande:

1. In quali giorni le autorità avrebbero dovuto emettere un'allerta 
   per la qualità dell'aria?

2. Quale stazione di monitoraggio sembra essere posizionata nella zona
   più calda della città?

3. C'è una correlazione tra temperatura alta e livelli di PM10?
   (Suggerimento: confronta i giorni con temp > 18°C e i livelli di PM10)

4. In base al trend di temperatura, come prevedi che sarà il giorno successivo?

5. Quali sono i vantaggi di usare NumPy rispetto alle liste Python standard
   per questo tipo di analisi?

===============================================================================
"""

# Funzione per verificare i risultati
def genera_report():
    """Genera un report riassuntivo delle analisi"""
    print("\n" + "="*70)
    print("REPORT SETTIMANALE - QUALITÀ DELL'ARIA E METEO")
    print("="*70)
    
    if temp_media is not None:
        print(f"\n📊 STATISTICHE TEMPERATURE:")
        print(f"   Media: {temp_media:.1f}°C")
        print(f"   Minima: {temp_minima:.1f}°C")
        print(f"   Massima: {temp_massima:.1f}°C")
        print(f"   Escursione: {escursione_termica:.1f}°C")
    
    if pm10_medio is not None:
        print(f"\n🏭 QUALITÀ DELL'ARIA (PM10):")
        print(f"   Media: {pm10_medio:.1f} μg/m³")
        print(f"   Giorni oltre soglia (50): {num_giorni_critici}")
        print(f"   Giorno peggiore: {giorno_peggiore + 1}")
    
    if giorni_pm10_alto is not None:
        giorni_alert = np.where(giorni_pm10_alto)[0] + 1
        if len(giorni_alert) > 0:
            print(f"   ⚠️  ALLERTA nei giorni: {giorni_alert}")
    
    print("\n" + "="*70)

# Decommentare per eseguire il report
# genera_report()