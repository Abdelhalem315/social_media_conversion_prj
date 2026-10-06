import joblib
import pandas as pd

model_path = 'models/xgboost_pipeline.joblib'

def load_artifacts(model_path = model_path):
    '''
    Loading Pipeline and model.
    '''
    print('Loading saved pipeline and model...')

    artifact = joblib.load(model_path)

    return artifact['preprocessing_pipeline'], artifact['model']

def predict_new_data(new_data: pd.DataFrame):
    '''
    Recieve new data
    predict the prediction
    '''
    pipeline, model = load_artifacts()

    # Transform new data...
    X_processed = pipeline.transform(new_data)

    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1] # For Marketing.

    results = new_data.copy()
    results['Predicted_Conversion'] = predictions
    results['Conversion_Probability'] = probabilities

    return results

if __name__ == "__main__":

    print('Testing prediction script on sample data...')

    raw_data = pd.read_csv('data/digital_marketing_campaign_dataset.csv')

    # try on 5 sample on data.
    sample_data = raw_data.sample(n=15)

    
    actual_conversion = sample_data['Conversion']
    features = sample_data.drop(columns=['Conversion'])

    
    output = predict_new_data(features)

    
    output['Actual_Conversion'] = actual_conversion

    # Performance Comparsion.. 
    print("--- Ground Truth vs Predictions ---")
    print(output[['Actual_Conversion', 'Predicted_Conversion', 'Conversion_Probability']])
