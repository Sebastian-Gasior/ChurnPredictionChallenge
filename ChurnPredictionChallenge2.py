# -------------------------------------------------------
# Import required packages
# -------------------------------------------------------
import pandas as pd
import numpy as np

# Machine Learning / Classification packages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Exakte Werte ausgeben
print(train_df['Churn'].value_counts(), "\n")

# Überprüfe fehlende Werte in den Datensätzen
print("Fehlende Werte in train_df:\n", train_df.isnull().sum())
print("Fehlende Werte in test_df:\n", test_df.isnull().sum())

# Zusätzliche Features in train_df und test_df erzeugen
train_df['TotalCharges_log'] = np.log1p(train_df['TotalCharges'])
test_df['TotalCharges_log'] = np.log1p(test_df['TotalCharges'])

# Neues Feature: durchschnittliche Kosten pro Monat (Vermeidung von Division durch Null)
train_df['ChargePerMonth'] = train_df['TotalCharges'] / (train_df['AccountAge'] + 1)
test_df['ChargePerMonth'] = test_df['TotalCharges'] / (test_df['AccountAge'] + 1)

# -------------------------------------------------------
# Data Preprocessing & Feature Engineering
# -------------------------------------------------------
# Trenne in Features und Zielvariable (aus train_df)
X_train = train_df.drop(['CustomerID', 'Churn'], axis=1)
y_train = train_df['Churn']

# Für test_df entferne 'CustomerID'
X_test = test_df.drop(['CustomerID'], axis=1)

# Kombiniere Trainings- und Testdaten, um konsistente Kodierung zu gewährleisten
combined = pd.concat([X_train, X_test], axis=0)

# Identifiziere numerische und kategoriale Spalten
num_cols = combined.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()

# Fülle fehlende numerische Werte (falls vorhanden) mit dem Median
imputer = SimpleImputer(strategy='median')
combined[num_cols] = imputer.fit_transform(combined[num_cols])

# One-Hot-Encoding für kategoriale Variablen (drop_first vermeidet Redundanz)
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

# Trenne die kombinierten Daten wieder in Trainings- und Testdaten
X_train_processed = combined_encoded.iloc[:len(X_train), :].copy()
X_test_processed = combined_encoded.iloc[len(X_train):, :].copy()

# Definiere den Parameter-Raum für das Tuning
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200]
}

# -------------------------------------------------------
# Modelltraining mit XGBoost
# -------------------------------------------------------
# Initialisiere den XGBoost-Klassifikator mit sinnvollen Hyperparametern
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Führe GridSearchCV mit 3-facher Kreuzvalidierung durch (du kannst cv anpassen)
grid_search = GridSearchCV(estimator=xgb_clf, 
                           param_grid=param_grid, 
                           scoring='roc_auc', 
                           cv=3, 
                           verbose=1, 
                           n_jobs=-1)

grid_search.fit(X_train_processed, y_train)

print("Beste Parameter:", grid_search.best_params_)
print("Bester ROC AUC Score (CV):", grid_search.best_score_)

# Verwende das beste Modell aus dem Grid Search
best_model = grid_search.best_estimator_
best_model.fit(X_train_processed, y_train)

# Berechne den ROC AUC auf den Trainingsdaten, um einen ersten Eindruck der Modellgüte zu erhalten
y_train_pred = best_model.predict_proba(X_train_processed)[:, 1]
auc_score = roc_auc_score(y_train, y_train_pred)
print("Train ROC AUC mit bestem Modell:", auc_score)

# -------------------------------------------------------
# Make predictions (required)
# -------------------------------------------------------
# Unser Modell wurde trainiert – jetzt machen wir Vorhersagen
predicted_probability = best_model.predict_proba(X_test_processed)[:, 1]

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