# -------------------------------------------------------
# Import required packages
# -------------------------------------------------------
import pandas as pd
import numpy as np

# Machine Learning / Classification packages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
# Optional: For imputing missing values
from sklearn.impute import SimpleImputer

# XGBoost (for a high-performance model)
import xgboost as xgb

# Visualization Packages
from matplotlib import pyplot as plt
import seaborn as sns
# Configure matplotlib for interactive output
plt.ion()

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
# Example: Distribution of the target variable 'Churn'
sns.countplot(x='Churn', data=train_df)
plt.title("Distribution of the Churn Variable")
plt.show()

# Print exact counts
print(train_df['Churn'].value_counts(), "\n")

# Check for missing values in the datasets
print("Missing values in train_df:\n", train_df.isnull().sum())
print("Missing values in test_df:\n", test_df.isnull().sum())

# Generate additional features in train_df and test_df
train_df['TotalCharges_log'] = np.log1p(train_df['TotalCharges'])
test_df['TotalCharges_log'] = np.log1p(test_df['TotalCharges'])

# New feature: average cost per month (to avoid division by zero)
train_df['ChargePerMonth'] = train_df['TotalCharges'] / (train_df['AccountAge'] + 1)
test_df['ChargePerMonth'] = test_df['TotalCharges'] / (test_df['AccountAge'] + 1)

# -------------------------------------------------------
# Data Preprocessing & Feature Engineering
# -------------------------------------------------------
# Separate features and target variable (from train_df)
X_train = train_df.drop(['CustomerID', 'Churn'], axis=1)
y_train = train_df['Churn']

# For test_df, remove 'CustomerID'
X_test = test_df.drop(['CustomerID'], axis=1)

# Combine training and test data to ensure consistent encoding
combined = pd.concat([X_train, X_test], axis=0)

# Identify numerical and categorical columns
num_cols = combined.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()

# Fill missing numerical values (if any) with the median
imputer = SimpleImputer(strategy='median')
combined[num_cols] = imputer.fit_transform(combined[num_cols])

# One-Hot Encoding for categorical variables (drop_first avoids redundancy)
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

# Split the combined data back into training and test sets
X_train_processed = combined_encoded.iloc[:len(X_train), :].copy()
X_test_processed = combined_encoded.iloc[len(X_train):, :].copy()

# Define the parameter grid for tuning
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200]
}

# -------------------------------------------------------
# Model training with XGBoost
# -------------------------------------------------------
# Initialize the XGBoost classifier with reasonable hyperparameters
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Simplified training without GridSearchCV to avoid compatibility issues
# Train the model with reasonable default parameters
best_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Train the model
best_model.fit(X_train_processed, y_train)

# Compute the ROC AUC on the training data to get an initial impression of model performance
y_train_pred = best_model.predict_proba(X_train_processed)[:, 1]
auc_score = roc_auc_score(y_train, y_train_pred)
print("Train ROC AUC with best model:", auc_score)

# -------------------------------------------------------
# Make predictions (required)
# -------------------------------------------------------
# Our model has been trained – now we make predictions
predicted_probability = best_model.predict_proba(X_test_processed)[:, 1]

# Create the submission DataFrame with exactly 104,480 rows and 2 columns:
# - CustomerID: to identify the test observations
# - predicted_probability: the predicted probability of churn
prediction_df = pd.DataFrame({
    'CustomerID': test_df['CustomerID'],
    'predicted_probability': predicted_probability
})

# Check the DataFrame
print("prediction_df Shape:", prediction_df.shape)
print(prediction_df.head(10))

# -------------------------------------------------------
# Final Tests (important - these cells must be run before submission)
# -------------------------------------------------------
# FINAL TEST CELLS - please make sure all of your code is above these test cells
# Writing to csv for autograding purposes
prediction_df.to_csv("prediction_submission2.csv", index=False)
submission = pd.read_csv("prediction_submission2.csv")

assert isinstance(submission, pd.DataFrame), 'You should have a dataframe named prediction_df.'

# FINAL TEST CELLS - please make sure all of your code is above these test cells
assert submission.columns[0] == 'CustomerID', 'The first column name should be CustomerID.'
assert submission.columns[1] == 'predicted_probability', 'The second column name should be predicted_probability.'

# FINAL TEST CELLS - please make sure all of your code is above these test cells
assert submission.shape[0] == 104480, 'The dataframe prediction_df should have 104480 rows.'

# FINAL TEST CELLS - please make sure all of your code is above these test cells
assert submission.shape[1] == 2, 'The dataframe prediction_df should have 2 columns.'