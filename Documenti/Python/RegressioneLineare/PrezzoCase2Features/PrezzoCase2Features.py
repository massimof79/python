import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # attiva il 3D

# -------------------------
# 1) DATI (simulati, realistici)
# -------------------------
np.random.seed(7)

n = 40
mq = np.random.uniform(45, 140, n)           # superficie
distanza = np.random.uniform(0.2, 4.0, n)    # km dalla metro

# Prezzo (in migliaia di euro): aumenta con mq, diminuisce con distanza + rumore
rumore = np.random.normal(0, 18, n)
prezzo = 2.6 * mq - 45 * distanza + 80 + rumore

# Feature matrix X e target y
X = np.column_stack([mq, distanza])
y = prezzo

# -------------------------
# 2) REGRESSIONE LINEARE
# -------------------------
model = LinearRegression()
model.fit(X, y)

w1, w2 = model.coef_
w0 = model.intercept_
r2 = model.score(X, y)

print("Modello: prezzo = w1*mq + w2*distanza + w0")
print(f"w1 (mq)       = {w1:.3f}")
print(f"w2 (distanza) = {w2:.3f}")
print(f"w0 (intercetta)= {w0:.3f}")
print(f"R^2 = {r2:.3f}")

# -------------------------
# 3) PREVISIONI DI ESEMPIO
# -------------------------
nuovi = np.array([
    [80, 0.5],   # 80 mq, 0.5 km
    [110, 3.2]   # 110 mq, 3.2 km
])
pred = model.predict(nuovi)

print("\nPrevisioni (in migliaia di euro):")
for (m, d), p in zip(nuovi, pred):
    print(f"- {m:.0f} mq, {d:.1f} km -> {p:.1f}")

# -------------------------
# 4) GRAFICO 3D: PUNTI + PIANO
# -------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# punti dati
ax.scatter(mq, distanza, prezzo)

# griglia per il piano
mq_grid = np.linspace(mq.min(), mq.max(), 20)
dist_grid = np.linspace(distanza.min(), distanza.max(), 20)
MQ, DIST = np.meshgrid(mq_grid, dist_grid)

# z del piano: usando l'equazione del modello
PREZZO_PLANO = w1 * MQ + w2 * DIST + w0

# disegna il piano
ax.plot_surface(MQ, DIST, PREZZO_PLANO, alpha=0.4)

ax.set_xlabel("Superficie (mq)")
ax.set_ylabel("Distanza metro (km)")
ax.set_zlabel("Prezzo (k€)")
ax.set_title("Regressione lineare con 2 feature: punti e piano")
plt.show()