# -------------------------------------------------------
# Import required packages
# -------------------------------------------------------
import pandas as pd
import numpy as np

# Machine Learning / Classification packages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer

# Ensemble and alternative models
from sklearn.ensemble import RandomForestClassifier

# Visualization Packages
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

# Import any other packages you may want to use
# XGBoost (for a high-performance model)
import xgboost as xgb

# Zelle 3
train_df = pd.read_csv("train.csv")
print('train_df Shape:', train_df.shape)
train_df.head()

# Zelle 4
test_df = pd.read_csv("test.csv")
print('test_df Shape:', test_df.shape)
test_df.head()

# Zelle 5
# Visualize the distribution of the target variable 'Churn'
sns.countplot(x='Churn', data=train_df)
plt.title("Distribution of the Churn Variable")
plt.show()
print(train_df['Churn'].value_counts(), "\n")

# Check for missing values in the datasets
print("Missing values in train_df:\n", train_df.isnull().sum(), "\n")
print("Missing values in test_df:\n", test_df.isnull().sum())

# Zelle 6
# Create additional features in train_df and test_df
# Data Preprocessing, Feature Engineering, and Outlier Handling

# Additional Feature Engineering:
# 1. Log transformation of TotalCharges to reduce skewness
train_df['TotalCharges_log'] = np.log1p(train_df['TotalCharges'])
test_df['TotalCharges_log'] = np.log1p(test_df['TotalCharges'])

# 2. Create a new feature: average charge per month (avoiding division by zero)
train_df['ChargePerMonth'] = train_df['TotalCharges'] / (train_df['AccountAge'] + 1)
test_df['ChargePerMonth'] = test_df['TotalCharges'] / (test_df['AccountAge'] + 1)

# 3. Create an interaction feature: product of AccountAge and MonthlyCharges
train_df['Age_MonthlyCharge'] = train_df['AccountAge'] * train_df['MonthlyCharges']
test_df['Age_MonthlyCharge'] = test_df['AccountAge'] * test_df['MonthlyCharges']

# Separate features and target from train_df
X_train = train_df.drop(['CustomerID', 'Churn'], axis=1)
y_train = train_df['Churn']

# For test_df, remove 'CustomerID'
X_test = test_df.drop(['CustomerID'], axis=1)

# Combine training and test data to ensure consistent encoding
combined = pd.concat([X_train, X_test], axis=0)

# Identify numerical and categorical columns
num_cols = combined.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()

# Outlier Handling: Clip numerical values using the IQR method
for col in num_cols:
    Q1 = combined[col].quantile(0.25)
    Q3 = combined[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    combined[col] = combined[col].clip(lower_bound, upper_bound)

# Fill missing numerical values (if any) with the median
imputer = SimpleImputer(strategy='median')
combined[num_cols] = imputer.fit_transform(combined[num_cols])

# One-Hot-Encoding for categorical variables (drop_first avoids redundancy)
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

# Split the combined data back into training and test sets
X_train_processed = combined_encoded.iloc[:len(X_train), :].copy()
X_test_processed = combined_encoded.iloc[len(X_train):, :].copy()

# Zelle 7
# Hyperparameter Tuning with GridSearchCV takes around 120 Min. to finish

# Advanced Hyperparameter Tuning for XGBoost Using RandomizedSearchCV

# Extended parameter grid for XGBoost
param_grid_extended = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

# Use StratifiedKFold for improved cross-validation (5 folds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up RandomizedSearchCV for XGBoost
random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_distributions=param_grid_extended,
    n_iter=50,  # number of random combinations to try
    scoring='roc_auc',
    cv=skf,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit the randomized search on the training data
random_search.fit(X_train_processed, y_train)

print("Best Parameters from Randomized Search:", random_search.best_params_)
print("Best ROC AUC Score (CV):", random_search.best_score_)

# Retrieve the best XGBoost model
best_xgb = random_search.best_estimator_

# Train an Alternative Model (RandomForest)
from sklearn.ensemble import RandomForestClassifier

# Initialize and train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_processed, y_train)

# Evaluate RandomForest on the training set
rf_train_pred = rf_model.predict_proba(X_train_processed)[:, 1]
rf_auc = roc_auc_score(y_train, rf_train_pred)
print("Random Forest Training ROC AUC:", rf_auc)

# Model Ensembling and Final Predictions
# Evaluate XGBoost on the training set for comparison
xgb_train_pred = best_xgb.predict_proba(X_train_processed)[:, 1]
xgb_auc = roc_auc_score(y_train, xgb_train_pred)
print("XGBoost Training ROC AUC:", xgb_auc)

# Create an ensemble by averaging the predicted probabilities from XGBoost and RandomForest
ensemble_train_pred = (xgb_train_pred + rf_train_pred) / 2
ensemble_train_auc = roc_auc_score(y_train, ensemble_train_pred)
print("Ensemble Training ROC AUC:", ensemble_train_auc)

# Generate predictions for the test set using the ensemble
xgb_test_pred = best_xgb.predict_proba(X_test_processed)[:, 1]
rf_test_pred = rf_model.predict_proba(X_test_processed)[:, 1]
ensemble_test_pred = (xgb_test_pred + rf_test_pred) / 2

# Create the submission DataFrame with exactly 104,480 rows and 2 columns:
# 'CustomerID' and 'predicted_probability'
prediction_df = pd.DataFrame({
    'CustomerID': test_df['CustomerID'],
    'predicted_probability': ensemble_test_pred
})

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