# credit_risk_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("cs-training.csv", index_col=0)

data.fillna(data.median(), inplace=True)

#feature Engineering
#for simplicity, we'll assume the dataset already has features like:
#'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
#'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
#'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
#'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'

#features and target
X = data.drop('SeriousDlqin2yrs', axis=1)
y = data['SeriousDlqin2yrs']  # 1 = default, 0 = non-default

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)

#xgboost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)

#evaluation of models
print("=== Random Forest Evaluation ===")
print(classification_report(y_test, rf_preds))
print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:,1]))
print()

print("=== XGBoost Evaluation ===")
print(classification_report(y_test, xgb_preds))
print("ROC AUC:", roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:,1]))

#flagging high risk customers
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df['PredictedDefault'] = xgb_preds
high_risk_customers = X_test_df[X_test_df['PredictedDefault'] == 1]
high_risk_customers.to_csv("flagged_high_risk_customers.csv", index=False)

