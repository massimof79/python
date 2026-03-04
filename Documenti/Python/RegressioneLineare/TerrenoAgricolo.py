import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# -------------------------
# 1) DATI (simulati)
# -------------------------
np.random.seed(7)

n = 70

fertilizzante = np.random.uniform(50, 250, n)   # kg per ettaro
pioggia = np.random.uniform(200, 700, n)        # mm stagionali
temperatura = np.random.uniform(12, 30, n)      # °C media

# rumore casuale dovuto a fattori non modellati
rumore = np.random.normal(0, 4, n)

# resa agricola (quintali/ettaro)
resa = 0.12 * fertilizzante + 0.015 * pioggia - 0.8 * temperatura + 40 + rumore

# matrice delle variabili indipendenti
X = np.column_stack([fertilizzante, pioggia, temperatura])
y = resa

# -------------------------
# 2) REGRESSIONE
# -------------------------
model = LinearRegression()
model.fit(X, y)

w1, w2, w3 = model.coef_
w0 = model.intercept_
r2 = model.score(X, y)

print("Modello: resa = w1*fertilizzante + w2*pioggia + w3*temperatura + w0")
print(f"w1 (fertilizzante) = {w1:.3f}")
print(f"w2 (pioggia)       = {w2:.3f}")
print(f"w3 (temperatura)   = {w3:.3f}")
print(f"w0 (intercetta)    = {w0:.3f}")
print(f"R^2 = {r2:.3f}")

# -------------------------
# 3) PREVISIONI
# -------------------------
nuovi = np.array([
    [120, 450, 18],
    [200, 300, 24]
])

pred = model.predict(nuovi)

print("\nPrevisioni resa (quintali/ettaro):")
for (f, p, t), r in zip(nuovi, pred):
    print(f"- fertilizzante {f:.0f} kg, pioggia {p:.0f} mm, temp {t:.1f} °C -> {r:.1f}")

# -------------------------
# 4) VISUALIZZAZIONE 3D
# Sezione con temperatura fissata
# -------------------------
temp_fissa = 20

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mask = np.abs(temperatura - temp_fissa) < 3
ax.scatter(fertilizzante[mask], pioggia[mask], resa[mask])

fert_grid = np.linspace(fertilizzante.min(), fertilizzante.max(), 25)
rain_grid = np.linspace(pioggia.min(), pioggia.max(), 25)

FERT, RAIN = np.meshgrid(fert_grid, rain_grid)

RESA_PLANO = w1 * FERT + w2 * RAIN + w3 * temp_fissa + w0

ax.plot_surface(FERT, RAIN, RESA_PLANO, alpha=0.4)

ax.set_xlabel("Fertilizzante (kg/ha)")
ax.set_ylabel("Pioggia (mm)")
ax.set_zlabel("Resa (quintali/ha)")
ax.set_title(f"Sezione del piano con temperatura = {temp_fissa} °C")

plt.show()