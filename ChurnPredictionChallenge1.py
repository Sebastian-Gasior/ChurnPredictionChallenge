# -------------------------------------------------------
# Import required packages
# -------------------------------------------------------
import pandas as pd
import numpy as np

# Machine Learning / Classification packages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# Optional: Für die Imputation fehlender Werte
from sklearn.impute import SimpleImputer

# XGBoost (für ein leistungsfähiges Modell)
import xgboost as xgb

# Visualization Packages
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

# -------------------------------------------------------
# Load the Data
# -------------------------------------------------------
train_df = pd.read_csv("train.csv")
print('train_df Shape:', train_df.shape)
train_df.head()
test_df = pd.read_csv("test.csv")
print('test_df Shape:', test_df.shape)
test_df.head()

# -------------------------------------------------------
# Explore, Clean, Validate, and Visualize the Data (optional)
# -------------------------------------------------------
# Beispiel: Verteilung der Zielvariable 'Churn'
sns.countplot(x='Churn', data=train_df)
plt.title("Verteilung der Churn-Variable")
plt.show()

# Fehlende Werte prüfen
print("Fehlende Werte in train_df:\n", train_df.isnull().sum())
print("Fehlende Werte in test_df:\n", test_df.isnull().sum())

# Falls 'TotalCharges' oder andere numerische Spalten als String vorliegen, 
# konvertiere sie in den entsprechenden Datentyp (hier als Beispiel übersprungen).

# -------------------------------------------------------
# Data Preprocessing & Feature Engineering
# -------------------------------------------------------
# Trenne in Features und Zielvariable (aus train_df)
X_train = train_df.drop(['CustomerID', 'Churn'], axis=1)
y_train = train_df['Churn']

# Für test_df entferne die Spalte 'CustomerID'
X_test = test_df.drop(['CustomerID'], axis=1)

# Kombiniere Trainings- und Testdaten, um eine konsistente Kodierung der kategorialen Features zu gewährleisten
combined = pd.concat([X_train, X_test], axis=0)

# Identifiziere numerische und kategoriale Spalten
num_cols = combined.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()

# Fülle fehlende numerische Werte mit dem Median
imputer = SimpleImputer(strategy='median')
combined[num_cols] = imputer.fit_transform(combined[num_cols])

# One-Hot-Encoding für kategoriale Variablen (drop_first vermeidet Redundanz)
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

# Trenne die kombinierten Daten wieder in Trainings- und Testdaten
X_train_processed = combined_encoded.iloc[:len(X_train), :].copy()
X_test_processed = combined_encoded.iloc[len(X_train):, :].copy()

# -------------------------------------------------------
# Modelltraining mit XGBoost
# -------------------------------------------------------
# Initialisiere den XGBoost-Klassifikator mit sinnvollen Hyperparametern
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Trainiere das Modell
model.fit(X_train_processed, y_train)

# Optional: Berechne den ROC AUC auf den Trainingsdaten, um einen ersten Eindruck der Modellgüte zu erhalten
y_train_pred = model.predict_proba(X_train_processed)[:, 1]
auc_score = roc_auc_score(y_train, y_train_pred)
print("Train ROC AUC:", auc_score)

# -------------------------------------------------------
# Make predictions (required)
# -------------------------------------------------------
# Unser Modell wurde trainiert – jetzt machen wir Vorhersagen
predicted_probability = model.predict_proba(X_test_processed)[:, 1]

# Erstelle den submission DataFrame mit exakt 104.480 Zeilen und 2 Spalten
# - CustomerID: zur Identifikation der Testbeobachtungen
# - predicted_probability: die vorhergesagte Wahrscheinlichkeit für Churn
prediction_df = pd.DataFrame({
    'CustomerID': test_df['CustomerID'],
    'predicted_probability': predicted_probability
})

# Überprüfe den DataFrame
print("prediction_df Shape:", prediction_df.shape)
print(prediction_df.head(10))

# -------------------------------------------------------
# Final Tests (wichtig - diese Zellen müssen vor der Einreichung ausgeführt werden)
# -------------------------------------------------------
# FINAL TEST CELLS - please make sure all of your code is above these test cells
# Writing to csv for autograding purposes
prediction_df.to_csv("prediction_submission.csv", index=False)
submission = pd.read_csv("prediction_submission.csv")

assert isinstance(submission, pd.DataFrame), 'You should have a dataframe named prediction_df.'

# FINAL TEST CELLS - please make sure all of your code is above these test cells
assert submission.columns[0] == 'CustomerID', 'The first column name should be CustomerID.'
assert submission.columns[1] == 'predicted_probability', 'The second column name should be predicted_probability.'

# FINAL TEST CELLS - please make sure all of your code is above these test cells
assert submission.shape[0] == 104480, 'The dataframe prediction_df should have 104480 rows.'

# FINAL TEST CELLS - please make sure all of your code is above these test cells
assert submission.shape[1] == 2, 'The dataframe prediction_df should have 2 columns.'