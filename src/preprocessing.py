from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

class ConstantColumnDropper(BaseEstimator, TransformerMixin):
    '''
    Custom Transformer to drop Zero-Variance / Constant Features
    '''
    def __init__(self, columns_to_drop= None): # None as a defualt value for safe.
        self.columns_to_drop = columns_to_drop if columns_to_drop else[]

    def fit(self, X, y= None): # I khnow we do NOT need this function here but it mandatory.
        return self 

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df.drop(columns=[c for c in self.columns_to_drop if c in X_df.columns])

class CustomMissingValueImputer(BaseEstimator, TransformerMixin):
    '''
    Defensive custom transformer to handle NaNs in both numerical 
    and categorical features for production readiness.
    Using Train set Only!
    '''
    def __init__(self, num_strategy= 'median', cat_strategy= 'most_frequent'):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.num_fill_values = {}
        self.cat_fill_values = {}

    def fit(self, X, y=None):
      X_df = pd.DataFrame(X).copy()

      # Learn fill values for numerical features.
      num_col = X_df.select_dtypes(include=[np.number]).copy()

      for c in num_col :
          if self.num_strategy == 'median':
              self.num_fill_values[c] = X_df[c].median()
          elif self.num_strategy == 'mean':
              self.num_fill_values[c] = X_df[c].mean()

        # Learn fill values for categorical features.
      cat_col = X_df.select_dtypes(include=['object', 'category']).copy()

      for c in cat_col :
          if self.cat_strategy == 'most_frequent':
              mode_value = X_df[c].mode()
              self.cat_fill_values[c] = mode_value[0] if not mode_value.empty else 'Unknown'
          elif self.cat_strategy == 'constant':
              self.cat_fill_values[c] = 'Unknown'
      return self

    def transform(self, X):
        '''
        Here applay atcual Imputation 
        for any new data (say test set).
        '''
        X_df = pd.DataFrame(X).copy()

        # Aplay numerical Imputation.
        for c, fill_value in self.num_fill_values.items():
            if c in X_df:
                X_df[c] = X_df[c].fillna(fill_value)

         # Aplay cateegorical Imputation.
        for c, fill_value in self.cat_fill_values.items():
            if c in X_df:
                X_df[c] = X_df[c].fillna(fill_value)
        return X_df

class IQROutlierCapper(BaseEstimator, TransformerMixin):
    '''
    Defensive custom transformers to cap outliers using Interqurtile range (IQR)
    upper_bound = q3 + factor * IQR
    lower_bound = q1 - factor * IQR
    '''
    def __init__(self, factor = 1.5):
        self.factor = factor
        self.upper_bound = {}
        self.lower_bound = {}

    def fit(self, X, y=None):
        # train the model to find the IQR (upper_bound, Lower_bound).
        X_df = pd.DataFrame(X).copy()

        num_col = X_df.select_dtypes(include=[np.number]).columns

        for col in num_col:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
        # Caculate absolute boundaries based on Training Data.
            self.upper_bound[col] = q3 + self.factor * iqr
            self.lower_bound[col] = q1 - self.factor * iqr

        return self

    def transform(self, X):
        '''
        Here apply Clipping
        per feature using
        np.clip method.
        '''
        X_df = pd.DataFrame(X).copy()

        for col, lower in self.lower_bound.items():
            if col in X_df:
                upper = self.upper_bound[col]
                X_df[col] = np.clip(X_df[col], lower, upper) # Clip Values to stay within [lower, upper] boundaries.
        return X_df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    '''
    Custom transformer to generate domain-specific interaction features (Composite metrix)
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()

        # 1. Ad Spend per Visit ratio
        if 'AdSpend' in X_df.columns and 'WebsiteVisits' in X_df.columns:
            X_df['AdSpend_per_Visit'] = X_df['AdSpend'] / (X_df['WebsiteVisits'] + 1e-5) 
        # 2. Overall Website Engagement Score
        if 'PagesPerVisit' in X_df.columns and 'TimeOnSite' in X_df.columns:
            X_df['Engagement_Score'] = X_df['PagesPerVisit'] * X_df['TimeOnSite']
        # 3. Loyalty relative to Income ratio
        if 'LoyaltyPoints' in X_df.columns and 'Income' in X_df.columns:
            X_df['Loyalty_per_Income'] = X_df['LoyaltyPoints'] / (X_df['Income'] + 1e-5)
        return X_df


# Pipeline Builder ...
def build_preprocessing_pipeline(
        constant_col = ['AdvertisingPlatform','AdvertisingTool'],
        num_col = [
            'Income',  # The Orginal Feauture that have a strong effect on target.
            'AdSpend', 
            'LoyaltyPoints',

            'AdSpend_per_Visit', # Composite Features For Strong Model.
            'Engagement_Score', 
            'Loyalty_per_Income'
        ],
        cat_col = ['Gender', 'CampaignChannel', 'CampaignType'] 
):
    """
    Constructs and returns the full unified Scikit-Learn preprocessing pipeline.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_col),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_col) 
        ],
        remainder='drop'
    )
    full_pipeline = Pipeline(steps=[
        ('drop_constants',  ConstantColumnDropper(columns_to_drop=constant_col)),
        ('imputer', CustomMissingValueImputer()),
        ('outlier_capper',IQROutlierCapper()),
        ('feature_engineering', FeatureEngineer()),
        ('encoding_and_scaling', column_transformer)
    ])

    return full_pipeline




        

        






              
    
          
