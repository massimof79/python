import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# -------------------------
# 1) DATI (simulati)
# -------------------------
np.random.seed(7)

n = 60
mq = np.random.uniform(45, 140, n)        # superficie
distanza = np.random.uniform(0.2, 4.0, n) # km metro
eta = np.random.uniform(0, 50, n)         # anni

# Prezzo (k€): cresce con mq, cala con distanza ed età + rumore
rumore = np.random.normal(0, 20, n)
prezzo = 2.5 * mq - 40 * distanza - 0.9 * eta + 90 + rumore

X = np.column_stack([mq, distanza, eta])
y = prezzo

# -------------------------
# 2) REGRESSIONE
# -------------------------
model = LinearRegression()
model.fit(X, y)

w1, w2, w3 = model.coef_
w0 = model.intercept_
r2 = model.score(X, y)

print("Modello: prezzo = w1*mq + w2*distanza + w3*eta + w0")
print(f"w1 (mq)        = {w1:.3f}")
print(f"w2 (distanza)  = {w2:.3f}")
print(f"w3 (eta)       = {w3:.3f}")
print(f"w0 (intercetta)= {w0:.3f}")
print(f"R^2 = {r2:.3f}")

# -------------------------
# 3) PREVISIONI
# -------------------------
nuovi = np.array([
    [80, 0.5, 5],    # 80 mq, 0.5 km, 5 anni
    [110, 3.2, 30]   # 110 mq, 3.2 km, 30 anni
])
pred = model.predict(nuovi)

print("\nPrevisioni (k€):")
for (m, d, e), p in zip(nuovi, pred):
    print(f"- {m:.0f} mq, {d:.1f} km, {e:.0f} anni -> {p:.1f}")

# -------------------------
# 4) VISUALIZZAZIONE 3D
# Sezione: età fissata (es. 20 anni)
# -------------------------
eta_fissa = 20

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# punti reali (mostriamo quelli con età vicina alla sezione)
mask = np.abs(eta - eta_fissa) < 5
ax.scatter(mq[mask], distanza[mask], prezzo[mask])

# griglia per il piano alla età fissata
mq_grid = np.linspace(mq.min(), mq.max(), 25)
dist_grid = np.linspace(distanza.min(), distanza.max(), 25)
MQ, DIST = np.meshgrid(mq_grid, dist_grid)

# z del piano con eta = eta_fissa
PREZZO_PLANO = w1 * MQ + w2 * DIST + w3 * eta_fissa + w0

ax.plot_surface(MQ, DIST, PREZZO_PLANO, alpha=0.4)

ax.set_xlabel("Superficie (mq)")
ax.set_ylabel("Distanza metro (km)")
ax.set_zlabel("Prezzo (k€)")
ax.set_title(f"Sezione del piano con età = {eta_fissa} anni")
plt.show()