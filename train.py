# The main objective: 'Automated Trainig Pipeline'.

import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from src.preprocessing import build_preprocessing_pipeline

def train():
    data_path = 'data/digital_marketing_campaign_dataset.csv'
    model_save_path = 'models/xgboost_pipeline.joblib' 

    print('Loading Data ....')
    data = pd.read_csv(data_path)

    X = data.drop(columns=['Conversion'])
    y = data['Conversion']

    ratio = (y == 0).sum() / (y == 1).sum() # calculate ratio of 'Imbalnce' to fix auto.

    preprocessing_pipeline = build_preprocessing_pipeline()
    X_processed = preprocessing_pipeline.fit_transform(X) # train model using all dataset.

    # Build and Train 'XGBoost' model using the best parameter.
    print('Training XGBoost model...')
    xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric='logloss'
 )
    xgb_model.fit(X_processed, y)


    '''
    Model Persistence ..
    The most important step
    Convert coding to production
    '''
    # fast check
    os.makedirs("models", exist_ok=True)

    artifact = {
        "preprocessing_pipeline": preprocessing_pipeline,
        "model": xgb_model
}

    # Save the file.
    joblib.dump(artifact, model_save_path)
    print(f'Model and Pipeline successfully saved to : {model_save_path}')

if __name__ == '__main__':
    train()