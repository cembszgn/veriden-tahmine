"""Veriden Tahmine — tekrar üretilebilir model ve etkileşimli mini tahmin aracı."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "veri_seti.csv"
FEATURES = ["bmi_standart", "kan_basinc_standart", "kan_biyobelirteci_s5"]
TARGET = "ilerleme_skoru"

def modeli_egit():
    df = pd.read_csv(DATA)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.20, random_state=42
    )
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("regressor", LinearRegression())])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"MAE": mean_absolute_error(y_test, pred),
               "RMSE": mean_squared_error(y_test, pred) ** 0.5,
               "R2": r2_score(y_test, pred)}
    return model, metrics, X_train, X_test, y_train, y_test

model, metrics, X_train, X_test, y_train, y_test = modeli_egit()
print(f"Eğitim: {X_train.shape} | Test: {X_test.shape}")
for name, value in metrics.items(): print(f"{name}: {value:.3f}")

def tahmin_et(bmi_standart, kan_basinc_standart, kan_biyobelirteci_s5):
    """Standartlaştırılmış 3 girdiyle ilerleme skorunu tahmin eder."""
    row = pd.DataFrame([dict(zip(FEATURES, [bmi_standart, kan_basinc_standart, kan_biyobelirteci_s5]))])
    return float(model.predict(row)[0])

if __name__ == "__main__":
    print("\nÖrnek tahmin:", round(tahmin_et(-0.05, -0.02, -0.03), 1))
    cevap = input("\nKendi girdilerinle tahmin yapmak ister misin? (e/h): ").strip().lower()
    if cevap == "e":
        try:
            bmi = float(input("Standartlaştırılmış BMI: "))
            bp = float(input("Standartlaştırılmış kan basıncı (bp): "))
            s5 = float(input("Standartlaştırılmış s5: "))
            print(f"Tahmin edilen ilerleme skoru: {tahmin_et(bmi, bp, s5):.1f}")
        except ValueError:
            print("Hata: Lütfen üç alan için de sayısal değer girin.")
