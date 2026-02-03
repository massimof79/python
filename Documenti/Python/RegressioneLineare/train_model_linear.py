import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Caricamento dataset Housing da OpenML
housing = fetch_openml(name="boston", version=1, as_frame=True)
df = housing.frame

features = ["RM", "LSTAT", "PTRATIO", "TAX", "AGE"]
X = df[features]
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Errore medio assoluto:", mean_absolute_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))

joblib.dump(pipeline, "modello_case.pkl")
print("Modello salvato in modello_case.pkl")
