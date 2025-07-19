import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBClassifier
import pickle
from preprocessing import CustomPreprocessor  # ✅ Clean import from separate file

# Load dataset
df = pd.read_excel("E Commerce Dataset.xlsx", sheet_name=1)
df.dropna(inplace=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Define categorical and numerical columns
categorical_cols = ['Gender', 'PreferredLoginDevice', 'PreferredPaymentMode',
                    'PreferedOrderCat', 'MaritalStatus']
numerical_cols = [col for col in X.columns if col not in categorical_cols and col != 'CustomerID']

# ColumnTransformer
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', MinMaxScaler(), numerical_cols)
])

# Pipeline
xgb_pipeline = Pipeline([
    ('cleanser', CustomPreprocessor()),
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Train
xgb_pipeline.fit(X, y)

# Save
with open('xgb_pipeline.pkl', 'wb') as f:
    pickle.dump(xgb_pipeline, f)

print("✅ Pipeline saved as xgb_pipeline.pkl")
