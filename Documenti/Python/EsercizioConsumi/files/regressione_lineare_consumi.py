"""
Esercizio: Previsione Consumi Energetici con Regressione Lineare
Autore: Massimo Farina
Descrizione: Modello di machine learning per prevedere i consumi energetici
             di un edificio scolastico utilizzando scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configurazione grafica
plt.style.use('seaborn-v0_8-darkgrid')

# Creazione cartella output se non esiste
OUTPUT_DIR = 'grafici_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✓ Cartella '{OUTPUT_DIR}' creata per salvare i grafici\n")

print("=" * 80)
print("PREVISIONE CONSUMI ENERGETICI - REGRESSIONE LINEARE")
print("=" * 80)
print()

# ============================================================================
# 1. CARICAMENTO E ANALISI ESPLORATIVA DEI DATI
# ============================================================================

print("1. CARICAMENTO DATASET")
print("-" * 80)

# Carica il dataset
df = pd.read_csv('consumi_energetici.csv')

print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
print("\nPrime righe del dataset:")
print(df.head(10))

print("\n\nStatistiche descrittive:")
print(df.describe())

print("\n\nInformazioni sul dataset:")
print(df.info())

print("\n\nVerifica valori mancanti:")
print(df.isnull().sum())

# ============================================================================
# 2. PREPARAZIONE DEI DATI
# ============================================================================

print("\n" + "=" * 80)
print("2. PREPARAZIONE DEI DATI")
print("-" * 80)

# Separazione features e target
X = df[['temperatura_media', 'giorni_lezione', 'studenti_presenti', 'ore_laboratori']]
y = df['consumo_kwh']

print(f"\nFeatures (X): {X.shape}")
print(f"Target (y): {y.shape}")

# Divisione train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")

# Normalizzazione (opzionale ma consigliata)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nDati normalizzati con StandardScaler")

# ============================================================================
# 3. ADDESTRAMENTO DEL MODELLO
# ============================================================================

print("\n" + "=" * 80)
print("3. ADDESTRAMENTO MODELLO DI REGRESSIONE LINEARE")
print("-" * 80)

# Creazione e addestramento del modello
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n✓ Modello addestrato con successo!")

# Visualizzazione coefficienti
print("\n\nCOEFFICIENTI DEL MODELLO:")
print("-" * 80)
print(f"Intercetta: {model.intercept_:.2f} kWh")
print("\nCoefficienti delle features:")

coefficients_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficiente': model.coef_
})
coefficients_df['Importanza_Assoluta'] = abs(coefficients_df['Coefficiente'])
coefficients_df = coefficients_df.sort_values('Importanza_Assoluta', ascending=False)

for idx, row in coefficients_df.iterrows():
    print(f"  {row['Feature']:25s}: {row['Coefficiente']:10.2f} kWh")

# Visualizzazione coefficienti
plt.figure(figsize=(10, 6))
bars = plt.barh(coefficients_df['Feature'], coefficients_df['Coefficiente'], 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
plt.xlabel('Coefficiente', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Importanza delle Features nel Modello', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/coefficienti_modello.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. PREVISIONI E VALUTAZIONE
# ============================================================================

print("\n" + "=" * 80)
print("4. VALUTAZIONE DELLE PRESTAZIONI")
print("-" * 80)

# Previsioni
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Metriche sul training set
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

print("\nMETRICHE SUL TRAINING SET:")
print(f"  R² Score:                {train_r2:.4f}")
print(f"  Mean Squared Error:      {train_mse:.2f}")
print(f"  Root Mean Squared Error: {train_rmse:.2f} kWh")
print(f"  Mean Absolute Error:     {train_mae:.2f} kWh")

# Metriche sul test set
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nMETRICHE SUL TEST SET:")
print(f"  R² Score:                {test_r2:.4f}")
print(f"  Mean Squared Error:      {test_mse:.2f}")
print(f"  Root Mean Squared Error: {test_rmse:.2f} kWh")
print(f"  Mean Absolute Error:     {test_mae:.2f} kWh")

# Interpretazione R²
print(f"\n✓ Il modello spiega il {test_r2*100:.1f}% della varianza nei consumi energetici")
print(f"✓ Errore medio di previsione: ±{test_mae:.0f} kWh")

# Visualizzazione previsioni vs valori reali
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training set
axes[0].scatter(y_train, y_train_pred, alpha=0.6, color='#4ECDC4', 
               s=100, edgecolors='black', linewidth=0.5)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
            'r--', lw=2, label='Previsione Perfetta')
axes[0].set_xlabel('Consumo Reale (kWh)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Consumo Previsto (kWh)', fontsize=11, fontweight='bold')
axes[0].set_title(f'Training Set - R²={train_r2:.3f}', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test set
axes[1].scatter(y_test, y_test_pred, alpha=0.6, color='#FF6B6B', 
               s=100, edgecolors='black', linewidth=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Previsione Perfetta')
axes[1].set_xlabel('Consumo Reale (kWh)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Consumo Previsto (kWh)', fontsize=11, fontweight='bold')
axes[1].set_title(f'Test Set - R²={test_r2:.3f}', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/previsioni_vs_reali.png', dpi=300, bbox_inches='tight')
plt.close()

# Residui (errori di previsione)
residui_train = y_train - y_train_pred
residui_test = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Distribuzione residui
axes[0].hist(residui_test, bins=10, color='#45B7D1', alpha=0.7, 
            edgecolor='black', linewidth=1.2)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Residui (kWh)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequenza', fontsize=11, fontweight='bold')
axes[0].set_title('Distribuzione dei Residui - Test Set', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Residui vs previsioni
axes[1].scatter(y_test_pred, residui_test, alpha=0.6, color='#FFA07A', 
               s=100, edgecolors='black', linewidth=0.5)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Consumo Previsto (kWh)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Residui (kWh)', fontsize=11, fontweight='bold')
axes[1].set_title('Residui vs Previsioni - Test Set', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/analisi_residui.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. PREVISIONI FUTURE
# ============================================================================

print("\n" + "=" * 80)
print("5. PREVISIONI PER I PROSSIMI MESI")
print("-" * 80)

# Dati stimati per i prossimi 3 mesi
nuovi_dati = pd.DataFrame({
    'temperatura_media': [8.5, 12.0, 16.5],
    'giorni_lezione': [18, 21, 20],
    'studenti_presenti': [860, 880, 870],
    'ore_laboratori': [325, 370, 355]
}, index=['Gennaio 2025', 'Febbraio 2025', 'Marzo 2025'])

print("\nDati di input per le previsioni:")
print(nuovi_dati)

# Normalizzazione e previsione
nuovi_dati_scaled = scaler.transform(nuovi_dati)
previsioni_future = model.predict(nuovi_dati_scaled)

print("\n\nPREVISIONI CONSUMO ENERGETICO:")
print("-" * 80)
for mese, consumo in zip(nuovi_dati.index, previsioni_future):
    print(f"  {mese:20s}: {consumo:8.0f} kWh")

print(f"\n✓ Consumo totale previsto: {previsioni_future.sum():,.0f} kWh")
print(f"✓ Consumo medio previsto: {previsioni_future.mean():,.0f} kWh/mese")

# ============================================================================
# 6. RIEPILOGO E CONCLUSIONI
# ============================================================================

print("\n" + "=" * 80)
print("6. RIEPILOGO E CONCLUSIONI")
print("=" * 80)

print("\n📊 RISULTATI PRINCIPALI:")
print("-" * 80)
print(f"✓ R² Score (Test):        {test_r2:.4f} ({test_r2*100:.1f}% varianza spiegata)")
print(f"✓ RMSE (Test):            {test_rmse:.2f} kWh")
print(f"✓ MAE (Test):             {test_mae:.2f} kWh")

print("\n🔍 FATTORI PIÙ INFLUENTI:")
print("-" * 80)
for idx, row in coefficients_df.head(3).iterrows():
    direzione = "aumenta" if row['Coefficiente'] > 0 else "riduce"
    print(f"  {idx+1}. {row['Feature']:20s} {direzione} i consumi")

print("\n💡 INTERPRETAZIONE:")
print("-" * 80)
if abs(coefficients_df.iloc[0]['Coefficiente']) > abs(coefficients_df.iloc[1]['Coefficiente']) * 2:
    print(f"  • La variabile '{coefficients_df.iloc[0]['Feature']}' ha un'influenza")
    print(f"    significativamente maggiore rispetto alle altre")
else:
    print(f"  • Le variabili hanno un'influenza relativamente equilibrata")

if test_r2 > 0.85:
    print(f"  • Il modello ha un'ottima capacità predittiva (R² > 0.85)")
elif test_r2 > 0.70:
    print(f"  • Il modello ha una buona capacità predittiva (R² > 0.70)")
else:
    print(f"  • Il modello potrebbe beneficiare di miglioramenti (R² < 0.70)")

print(f"\n✅ TUTTI I GRAFICI SONO STATI SALVATI NELLA CARTELLA '{OUTPUT_DIR}/'")
print("\n" + "=" * 80)
