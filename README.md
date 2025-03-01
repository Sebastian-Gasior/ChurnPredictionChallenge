# Churn Prediction Challenge

This project is based on a real-world coding challenge from Coursera. In this challenge, you are tasked with predicting subscription churn for a video streaming service using historical subscription data. The challenge provides two datasets:

- `train.csv`: Contains 70% of the overall sample (243,787 subscriptions) with the ground truth for whether a subscription was continued (the target label Churn).
- `test.csv`: Contains the remaining 30% (104,480 subscriptions) without the target label. Your job is to predict the likelihood of churn for each test entry.

Link: https://www.coursera.org/projects/data-science-challenge

## Features

Data Exploration and Visualization:
Explore the distribution of the target variable (Churn), check for missing values, and inspect the dataset.

Data Preprocessing & Feature Engineering:
Various scripts apply different approaches to data cleaning, handling outliers, one-hot encoding of categorical variables, and creating new features (e.g., log transformations, interaction features).

Model Training and Evaluation:
Different machine learning models are employed (primarily XGBoost and RandomForest) and evaluated using the ROC AUC metric.

Model Ensembling:
Some scripts combine multiple models by averaging predicted probabilities to enhance performance.

Automated Submission Format:
Final test cells ensure that the generated submission file meets the required format (a CSV with exactly 104,480 rows and 2 columns: CustomerID and predicted_probability).

## Installation

To run this project locally, please follow these steps:

1. Clone the Repository
Clone the project repository to your local machine.

2. Create a Virtual Environment
Create and activate a virtual environment (using venv):
Windows (PowerShell):
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you encounter an execution policy error, temporarily run:

Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies
Install the required packages using:
```bash
pip install -r requirements.txt
```

Note: If you encounter compiler issues during installation (especially with pandas), ensure that you have Microsoft Visual C++ Build Tools installed and consider upgrading pip, setuptools, and wheel:
```bash
python -m pip install --upgrade pip setuptools wheel
```

After submission, your performance is evaluated by comparing your predicted probabilities (using the ROC AUC metric) against the true outcomes.

## Project Structure

The project contains four Python scripts, each representing a different approach to solving the churn prediction challenge:

**ChurnPredictionChallenge1.py**
Approach: Uses a basic XGBoost model with minimal feature engineering and no advanced hyperparameter tuning.
Result: Train ROC AUC: 0.7759
Advantages: Simplicity and ease of implementation.

**ChurnPredictionChallenge2.py**
Approach: Implements hyperparameter tuning using GridSearchCV to optimize an XGBoost model.
Result: Train ROC AUC with best model: 0.7646
Advantages: Systematic parameter optimization, though it may slightly lower the training ROC AUC due to tuning trade-offs.

**ChurnPredictionChallenge3.py**
Approach: Trains both RandomForest and XGBoost models, then ensembles them by averaging predictions.
Results:
    Random Forest Training ROC AUC: 0.7499
    XGBoost Training ROC AUC: 0.7900
    Ensemble Training ROC AUC: 0.7791
Advantages: Demonstrates improved performance through ensembling.

**ChurnPredictionChallenge4.py**
Approach: Similar to Challenge 3, this script also uses both RandomForest and XGBoost with an ensemble method.
Results: Identical to Challenge 3:
    Random Forest Training ROC AUC: 0.7499
    XGBoost Training ROC AUC: 0.7900
    Ensemble Training ROC AUC: 0.7791
Advantages: Reinforces the ensemble approach and validates its consistency.

## Packages and Dependencies

The project relies on the following key packages:

- pandas and numpy: For data manipulation and numerical operations.
- scikit-learn: For machine learning, including model training, evaluation (ROC AUC), data preprocessing, and hyperparameter tuning.
- xgboost: For training high-performance gradient boosting models.
- matplotlib and seaborn: For data visualization.
- joblib: For model persistence and parallel processing (if needed).

All dependencies are listed in the requirements.txt file.

## Execution

The project is primarily developed in Jupyter Notebook, but the provided Python scripts can be run locally. Each script performs the following steps:

1. Load Data: Reads train.csv and test.csv.
2. Data Exploration: Visualizes and checks data quality.
3. Data Preprocessing & Feature Engineering: Processes the data, handles outliers, and creates additional features.
4. Model Training & Evaluation: Trains models (XGBoost, RandomForest, or ensembles) and computes the ROC AUC on the training set.
5. Prediction and Submission: Generates predictions for the test set and outputs a submission file (prediction_submission.csv) in the required format.
6. Final Tests: Runs final test cells to validate the submission format.

To run a script, activate your virtual environment and execute:
```bash
python ChurnPredictionChallengeX.py
```

Replace X with 1, 2, 3, or 4.

## Development

This project was developed as part of a Coursera coding challenge. During development, several approaches were experimented with:

- Data Exploration & Cleaning: Thoroughly examined data quality and handled missing values/outliers.
- Feature Engineering: Created new features like log-transformed TotalCharges, ChargePerMonth, and interaction terms.
- Hyperparameter Tuning: Tested methods like GridSearchCV and RandomizedSearchCV for optimizing model parameters.
- Multiple Algorithms: Evaluated different models (XGBoost, RandomForest) and combined them via ensembling.
- Ensembling: Averaged predictions from different models to enhance performance.
- Iterative Improvement: Compared training ROC AUC scores for each approach to determine the most effective solution.

The iterative process led to four distinct scripts, each highlighting various techniques and trade-offs.


## Known Issues & Solutions

Compiler Issues during Installation:
If you encounter errors related to the C/C++ compiler (e.g., when installing pandas), ensure that Microsoft Visual C++ Build Tools are installed and your pip, setuptools, and wheel packages are up to date.

Virtual Environment Activation Issues:
On Windows, if you receive execution policy errors when activating the virtual environment in PowerShell, use:
```bash
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

Then activate using:
```bash
.\venv\Scripts\Activate.ps1
```

Differences Between Jupyter and Local Execution:
Although the code is developed in a Jupyter Notebook environment, minor adjustments (e.g., using plt.ion() for interactive plotting) may be necessary when running the scripts locally.


DEUTSCH

# Churn Prediction Challenge

Dieses Projekt basiert auf einer realen Coding-Challenge von Coursera. In dieser Challenge besteht Ihre Aufgabe darin, die Abwanderung (Churn) von Abonnenten eines Video-Streaming-Dienstes anhand historischer Abonnementdaten vorherzusagen. Die Challenge stellt zwei Datensätze zur Verfügung:

- `train.csv`: Enthält 70 % der Gesamtheit (genau 243.787 Abonnements) und liefert die tatsächlichen Werte, ob ein Abonnement verlängert wurde (das Zielattribut Churn).
- `test.csv`: Enthält die restlichen 30 % (genau 104.480 Abonnements) ohne Zielattribut. Ihre Aufgabe ist es, die Wahrscheinlichkeit des Churns für jeden Eintrag im Testdatensatz vorherzusagen.

Nach der Einreichung wird Ihre Leistung anhand des ROC AUC (Receiver Operating Characteristic Area Under the Curve) bewertet, indem Ihre vorhergesagten Wahrscheinlichkeiten mit den tatsächlichen Werten verglichen werden.

Link: https://www.coursera.org/projects/data-science-challenge

## Features

Datenexploration und -visualisierung:
Untersuchen Sie die Verteilung des Zielattributs (Churn), überprüfen Sie auf fehlende Werte und analysieren Sie den Datensatz.

Datenvorverarbeitung & Feature Engineering:
Verschiedene Skripte wenden unterschiedliche Ansätze zur Datenbereinigung, zum Umgang mit Ausreißern, zur One-Hot-Kodierung von kategorialen Variablen und zur Erstellung neuer Features an (z. B. Log-Transformationen, Interaktionsfeatures).

Modelltraining und -bewertung:
Es werden verschiedene Machine-Learning-Modelle eingesetzt (hauptsächlich XGBoost und RandomForest), die anhand des ROC AUC bewertet werden.

Modell-Ensembling:
Einige Skripte kombinieren mehrere Modelle, indem die vorhergesagten Wahrscheinlichkeiten gemittelt werden, um die Leistung zu verbessern.

Automatisiertes Einreichungsformat:
Abschließende Testzellen stellen sicher, dass die erzeugte Einreichungsdatei das erforderliche Format (eine CSV mit genau 104.480 Zeilen und 2 Spalten: CustomerID und predicted_probability) erfüllt.

## Installation

Um dieses Projekt lokal auszuführen, befolgen Sie bitte die folgenden Schritte:

1. Repository klonen
    Klonen Sie das Projekt-Repository auf Ihren lokalen Rechner.

2. Virtuelle Umgebung erstellen
Erstellen und aktivieren Sie eine virtuelle Umgebung (z. B. mit venv):
Windows (PowerShell):
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Falls Sie eine Ausführungsrichtlinien-Fehlermeldung erhalten, führen Sie temporär aus:
```bash
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Abhängigkeiten installieren
Installieren Sie die benötigten Pakete mit:
```bash
pip install -r requirements.txt
```

Hinweis: Sollten Sie Compiler-Probleme während der Installation (insbesondere bei pandas) haben, stellen Sie sicher, dass Sie die Microsoft Visual C++ Build Tools installiert haben, und aktualisieren Sie pip, setuptools und wheel:
```bash
python -m pip install --upgrade pip setuptools wheel
```

## Projektstruktur

Das Projekt enthält vier Python-Skripte, die jeweils einen anderen Ansatz zur Lösung der Churn Prediction Challenge darstellen:

**ChurnPredictionChallenge1.py**
    Ansatz: Verwendet ein einfaches XGBoost-Modell mit minimalem Feature Engineering und ohne fortgeschrittenes Hyperparameter-Tuning.
    Ergebnis: Train ROC AUC: 0.7759
    Vorteile: Einfachheit und leichte Implementierung.

**ChurnPredictionChallenge2.py**
    Ansatz: Implementiert Hyperparameter-Tuning mittels GridSearchCV zur Optimierung eines XGBoost-Modells.
    Ergebnis: Train ROC AUC mit bestem Modell: 0.7646
    Vorteile: Systematische Parameteroptimierung, obwohl dies zu einem leichten Abfall des Trainings-ROC AUC führen kann.

**ChurnPredictionChallenge3.py**
    Ansatz: Trainiert sowohl RandomForest- als auch XGBoost-Modelle und ensemblet diese durch Mittelung der Vorhersagen.
    Ergebnisse:
        Random Forest Training ROC AUC: 0.7499
        XGBoost Training ROC AUC: 0.7900
        Ensemble Training ROC AUC: 0.7791
    Vorteile: Zeigt verbesserte Leistung durch Ensembling.

**ChurnPredictionChallenge4.py**
    Ansatz: Ähnlich wie Challenge 3, verwendet dieses Skript ebenfalls RandomForest und XGBoost mit einer Ensemble-Methode.
    Ergebnisse: Identisch zu Challenge 3:
        Random Forest Training ROC AUC: 0.7499
        XGBoost Training ROC AUC: 0.7900
        Ensemble Training ROC AUC: 0.7791
    Vorteile: Bestätigt den Erfolg des Ensemble-Ansatzes und seine Konsistenz.

## Pakete und Abhängigkeiten

Das Projekt basiert auf folgenden wichtigen Paketen:

    - pandas und numpy: Für die Datenmanipulation und numerische Operationen.
    - scikit-learn: Für Machine Learning, einschließlich Modelltraining, Bewertung (ROC AUC), Datenvorverarbeitung und Hyperparameter-Tuning.
    - xgboost: Zum Training leistungsstarker Gradient-Boosting-Modelle.
    - matplotlib und seaborn: Für Datenvisualisierung.
    - joblib: Für Modelldefinition und parallele Verarbeitung (falls erforderlich).

Alle Abhängigkeiten sind in der Datei requirements.txt aufgeführt.

## Ausführung

Das Projekt wurde hauptsächlich in Jupyter Notebook entwickelt, aber die bereitgestellten Python-Skripte können auch lokal ausgeführt werden. Jedes Skript führt die folgenden Schritte durch:

1. Daten laden: Liest train.csv und test.csv.
2. Datenexploration: Visualisiert und überprüft die Datenqualität.
3. Datenvorverarbeitung & Feature Engineering: Verarbeitet die Daten, behandelt Ausreißer und erstellt zusätzliche Features.
4. Modelltraining & Bewertung: Trainiert Modelle (XGBoost, RandomForest oder Ensembles) und berechnet den ROC AUC auf den Trainingsdaten.
5. Vorhersage und Einreichung: Generiert Vorhersagen für den Testdatensatz und erstellt eine Einreichungsdatei (prediction_submission.csv) im erforderlichen Format.
6. Finale Tests: Führt abschließende Testzellen aus, um das Einreichungsformat zu validieren.

Um ein Skript auszuführen, aktivieren Sie Ihre virtuelle Umgebung und führen Sie folgenden Befehl aus:
```bash
python ChurnPredictionChallengeX.py
```

Ersetzen Sie X durch 1, 2, 3 oder 4.

## Entwicklung

Dieses Projekt wurde im Rahmen einer Coursera-Coding-Challenge entwickelt. Während der Entwicklung wurden mehrere Ansätze ausprobiert:

Datenexploration & Bereinigung:
Die Daten wurden gründlich untersucht, um die Datenqualität zu bewerten und fehlende Werte bzw. Ausreißer zu behandeln.

Feature Engineering:
Neue Features wurden erstellt, wie z. B. log-transformierte TotalCharges, ChargePerMonth und Interaktionstermen, um zusätzliche Informationen zu erfassen.

Hyperparameter-Tuning:
Methoden wie GridSearchCV und RandomizedSearchCV wurden getestet, um die Modellparameter zu optimieren.

Mehrere Algorithmen:
Unterschiedliche Modelle (XGBoost, RandomForest) wurden evaluiert und kombiniert, um die beste Vorhersageleistung zu erzielen.

Ensembling:
Die Vorhersagen mehrerer Modelle wurden gemittelt, um die Gesamtleistung zu verbessern.

Iterative Verbesserung:
Die Trainings-ROC AUC-Werte der verschiedenen Ansätze wurden verglichen, um den effektivsten Lösungsweg zu identifizieren.

Dieser iterative Prozess führte zu vier verschiedenen Skripten, die jeweils verschiedene Techniken und Kompromisse aufzeigen.
    
## Bekannte Probleme & Lösungen

Compiler-Probleme während der Installation:
Falls Fehler im Zusammenhang mit dem C/C++-Compiler (z. B. bei der Installation von pandas) auftreten, stellen Sie sicher, dass die Microsoft Visual C++ Build Tools installiert sind, und aktualisieren Sie pip, setuptools und wheel.

Probleme beim Aktivieren der virtuellen Umgebung:
Auf Windows kann es zu Ausführungsrichtlinien-Fehlermeldungen kommen. Verwenden Sie in PowerShell:
```bash
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

und aktivieren Sie anschließend die Umgebung mit:
```bash
.\venv\Scripts\Activate.ps1
```

Unterschiede zwischen Jupyter und lokaler Ausführung:
Obwohl der Code in einer Jupyter Notebook-Umgebung entwickelt wurde, können geringfügige Anpassungen (z. B. die Verwendung von plt.ion() für interaktives Plotting) erforderlich sein, wenn Sie die Skripte lokal ausführen.