from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    Cleans up inconsistent values before encoding.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Drop ID column if present
        if 'CustomerID' in X.columns:
            X = X.drop('CustomerID', axis=1)

        # Normalize value inconsistencies
        X['PreferredLoginDevice'] = X['PreferredLoginDevice'].replace('Phone', 'Mobile Phone')
        X['PreferredPaymentMode'] = X['PreferredPaymentMode'].replace({'COD': 'Cash on Delivery', 'CC': 'Credit Card'})
        X['PreferedOrderCat'] = X['PreferedOrderCat'].replace('Mobile', 'Mobile Phone')

        return X
