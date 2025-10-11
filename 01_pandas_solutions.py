# SOLUZIONI - Esercizio Pandas: Cambiamento Climatico

import pandas as pd
import numpy as np

# Creazione dei dataset
temperature = {
    'Anno': range(1990, 2024),
    'Temperatura': np.random.uniform(14.2, 15.1, 34),
    'Anomalia': np.random.uniform(0.3, 1.2, 34),
    'Continente': np.random.choice(['Europa', 'Asia', 'America', 'Africa', 'Oceania'], 34)
}
#Creo un dataframe per le temperature
df_temp = pd.DataFrame(temperature)

emissioni = {
    'Paese': ['Cina', 'USA', 'India', 'Russia', 'Giappone', 'Germania', 'Iran', 'Indonesia'],
    'CO2_milioni_ton': [10065, 5416, 2654, 1711, 1162, 759, 720, 615],
    'Popolazione_milioni': [1412, 331, 1380, 144, 126, 83, 84, 274],
    'PIL_procapite': [12556, 69375, 2389, 11654, 39285, 51203, 3877, 4357]
}
df_emissioni = pd.DataFrame(emissioni)

energie_rinnovabili = {
    'Paese': ['Cina', 'USA', 'Germania', 'Giappone', 'India'],
    'Percentuale_rinnovabili': [28.8, 20.1, 41.1, 22.4, 24.3]
}
df_rinnovabili = pd.DataFrame(energie_rinnovabili)

# ============================================================================
# PARTE 1: CREAZIONE E ESPLORAZIONE BASE
# ============================================================================

# Esercizio 1: Visualizza le prime 10 righe
print("ESERCIZIO 1:")
print(df_temp.head(10))
print("\n" + "="*80 + "\n")

# Esercizio 2: Informazioni sulla struttura
print("ESERCIZIO 2:")
print(df_temp.info())
print("\n" + "="*80 + "\n")

# Esercizio 3: Statistiche descrittive
print("ESERCIZIO 3:")
print(df_temp.describe())
print("\n" + "="*80 + "\n")

# Esercizio 4: Verifica valori mancanti
print("ESERCIZIO 4:")
print(df_temp.isnull().sum())
# oppure
print(df_temp.isna().sum())
print("\n" + "="*80 + "\n")

# ============================================================================
# PARTE 2: SELEZIONE E FILTRO
# ============================================================================

# Esercizio 5: Seleziona colonna Temperatura
print("ESERCIZIO 5:")
temperatura_col = df_temp['Temperatura']
# oppure
temperatura_col = df_temp.Temperatura
print(temperatura_col)
print("\n" + "="*80 + "\n")

# Esercizio 6: Seleziona anni 2010-2020
print("ESERCIZIO 6:")
df_2010_2020 = df_temp[(df_temp['Anno'] >= 2010) & (df_temp['Anno'] <= 2020)]
print(df_2010_2020)
print("\n" + "="*80 + "\n")

# Esercizio 7: Filtra anomalia > 0.8
print("ESERCIZIO 7:")
df_alta_anomalia = df_temp[df_temp['Anomalia'] > 0.8]
print(df_alta_anomalia)
print("\n" + "="*80 + "\n")

# Esercizio 8: Temperatura > 14.8 in Europa
print("ESERCIZIO 8:")
df_europa_caldo = df_temp[(df_temp['Temperatura'] > 14.8) & (df_temp['Continente'] == 'Europa')]
print(df_europa_caldo)
print("\n" + "="*80 + "\n")

# ============================================================================
# PARTE 3: ORDINAMENTO E RANKING
# ============================================================================

# Esercizio 9: Ordina per temperatura decrescente
print("ESERCIZIO 9:")
df_ordinato = df_temp.sort_values('Temperatura', ascending=False)
print(df_ordinato)
print("\n" + "="*80 + "\n")

# Esercizio 10: I 5 anni più caldi
print("ESERCIZIO 10:")
top_5_caldi = df_temp.nlargest(5, 'Temperatura')
print(top_5_caldi)
print("\n" + "="*80 + "\n")

# Esercizio 11: Crea colonna Ranking
print("ESERCIZIO 11:")
df_temp_sorted = df_temp.sort_values('Temperatura', ascending=False)
df_temp_sorted['Ranking'] = range(1, len(df_temp_sorted) + 1)
print(df_temp_sorted)
print("\n" + "="*80 + "\n")

# ============================================================================
# PARTE 4: OPERAZIONI SUI DATI
# ============================================================================

# Esercizio 12: Calcola emissioni pro capite
print("ESERCIZIO 12:")
df_emissioni['Emissioni_procapite'] = df_emissioni['CO2_milioni_ton'] / df_emissioni['Popolazione_milioni']
print(df_emissioni)
print("\n" + "="*80 + "\n")

# Esercizio 13: Aggiungi colonna Categoria
print("ESERCIZIO 13:")
def categorizza_emissioni(valore):
    if valore > 10:
        return 'Alto'
    elif valore >= 5:
        return 'Medio'
    else:
        return 'Basso'

df_emissioni['Categoria'] = df_emissioni['Emissioni_procapite'].apply(categorizza_emissioni)
print(df_emissioni)
print("\n" + "="*80 + "\n")

# Esercizio 14: Media emissioni per categoria
print("ESERCIZIO 14:")
media_per_categoria = df_emissioni.groupby('Categoria')['CO2_milioni_ton'].mean()
print(media_per_categoria)
print("\n" + "="*80 + "\n")

# ============================================================================
# PARTE 5: RAGGRUPPAMENTO E AGGREGAZIONE
# ============================================================================

# Esercizio 15: Temperatura media per continente
print("ESERCIZIO 15:")
temp_media_continente = df_temp.groupby('Continente')['Temperatura'].mean()
print(temp_media_continente)
print("\n" + "="*80 + "\n")

# Esercizio 16: Anno con anomalia massima per continente
print("ESERCIZIO 16:")
anno_max_anomalia = df_temp.loc[df_temp.groupby('Continente')['Anomalia'].idxmax()]
print(anno_max_anomalia[['Continente', 'Anno', 'Anomalia']])
print("\n" + "="*80 + "\n")

# Esercizio 17: Tabella pivot per decenni
print("ESERCIZIO 17:")
# Crea colonna Decennio
def assegna_decennio(anno):
    if anno < 2000:
        return '1990-1999'
    elif anno < 2010:
        return '2000-2009'
    elif anno < 2020:
        return '2010-2019'
    else:
        return '2020-2023'

df_temp['Decennio'] = df_temp['Anno'].apply(assegna_decennio)

# Crea pivot table
pivot = df_temp.pivot_table(values='Anomalia', 
                             index='Continente', 
                             columns='Decennio', 
                             aggfunc='mean')
print(pivot)
print("\n" + "="*80 + "\n")

# ============================================================================
# PARTE 6: JOIN E MERGE
# ============================================================================

# Esercizio 18: Merge dei DataFrame
print("ESERCIZIO 18:")
df_completo = df_emissioni.merge(df_rinnovabili, on='Paese', how='left')
print(df_completo)
print("\n" + "="*80 + "\n")

# Esercizio 19: Analisi relazione
print("ESERCIZIO 19:")
# Filtra paesi con dati completi
df_analisi = df_completo.dropna(subset=['Percentuale_rinnovabili'])

# Visualizza correlazione
print(df_analisi[['Emissioni_procapite', 'Percentuale_rinnovabili']])
print("\nCorrelazione:", 
      df_analisi['Emissioni_procapite'].corr(df_analisi['Percentuale_rinnovabili']))
print("\n" + "="*80 + "\n")

# ============================================================================
# PARTE 7: PULIZIA E TRASFORMAZIONE
# ============================================================================

# Esercizio 20: Sostituisci Oceania
print("ESERCIZIO 20:")
df_temp['Continente'] = df_temp['Continente'].replace('Oceania', 'Australia-Oceania')
print(df_temp['Continente'].unique())
print("\n" + "="*80 + "\n")

# Esercizio 21: Rimuovi duplicati
print("ESERCIZIO 21:")
df_temp_pulito = df_temp.drop_duplicates()
print(f"Righe prima: {len(df_temp)}, Righe dopo: {len(df_temp_pulito)}")
print("\n" + "="*80 + "\n")

# Esercizio 22: Reset indice dopo filtro
print("ESERCIZIO 22:")
df_europa = df_temp[df_temp['Continente'] == 'Europa'].reset_index(drop=True)
print(df_europa)
print("\n" + "="*80 + "\n")

# ============================================================================
# DOMANDE DI RIFLESSIONE
# ============================================================================

print("RISPOSTE ALLE DOMANDE DI RIFLESSIONE:")
print("="*80)

# Continenti con anomalie più elevate
print("\nAnomalia media per continente:")
print(df_temp.groupby('Continente')['Anomalia'].mean().sort_values(ascending=False))

# Correlazione PIL-Emissioni
print("\nCorrelazione PIL pro capite - Emissioni pro capite:")
print(df_emissioni['PIL_procapite'].corr(df_emissioni['Emissioni_procapite']))

# Confronto decenni
primo_decennio = df_temp[df_temp['Anno'] < 2000]['Temperatura'].mean()
ultimo_decennio = df_temp[df_temp['Anno'] >= 2014]['Temperatura'].mean()
print(f"\nTemperatura media 1990-1999: {primo_decennio:.2f}°C")
print(f"Temperatura media 2014-2023: {ultimo_decennio:.2f}°C")
print(f"Differenza: {ultimo_decennio - primo_decennio:.2f}°C")