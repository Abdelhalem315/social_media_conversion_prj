# End-to-End Modular Machine Learning Pipeline
## Social Media Marketing Conversion Predictor

### Executive Summary & Business Problem
In digital marketing, allocating ad budgets efficiently requires understanding which customer segments and acquisition channels yield the highest conversion rates. This project delivers a production-ready machine learning pipeline that predicts whether a user visiting a website from a specific marketing campaign will result in a successful conversion. By providing real-time conversion probabilities, the system enables marketing teams to optimize ad spend, target high-intent customer profiles, and maximize return on ad spend (ROAS).

### Architectural Approach & Engineering Workflow
The core architecture follows a decoupled, modular design pattern built for reproducibility and deployment readiness:

1. **Problem Formulation & Data Splitting**: To prevent data leakage, raw dataset splits (Train/Test) were established prior to exploratory data analysis (EDA) and feature engineering.
2. **Exploratory Data Analysis (EDA)**: Conducted to analyze feature distributions, correlations, and target class imbalances (87.6% non-conversions vs. 12.4% conversions).
3. **Custom Preprocessing Pipeline**: Built reusable scikit-learn compatible transformers (`src/preprocessing.py`) to handle out-of-bounds outliers via Interquartile Range (IQR) clipping and standardized scaling.
4. **Iterative Model Selection**:
   * **Baseline Model (Logistic Regression)**: Established initial performance metrics; struggled with non-linear relationships and high class imbalance.
   * **Tree-Based Baseline (Random Forest)**: Improved predictive power after applying regularization constraints to prevent overfitting.
   * **Final Production Model (XGBoost)**: Achieved optimal precision-recall trade-offs and gradient boosting performance across imbalanced target classes.
5. **Pipeline Artifact Persistence**: Fitted transformers and the final XGBoost estimator are serialized into a single binary file (`models/xgboost_pipeline.joblib`) for unified inference.

### Deployment & Interface Access
The trained pipeline can be executed via two primary interfaces:
* **Command Line Interface (CLI)**: Orchestrated via `main.py` for automated training, batch evaluations, and test predictions.
* **Interactive Dashboard**: A lightweight Streamlit application (`app.py`) for real-time inference and user-input predictions.

### Repository Structure & Modular Design

```text
social_media_conversion_prj/
├── data/
│   └── digital_marketing_campaign_dataset.csv
├── models/
│   └── xgboost_pipeline.joblib
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── __init__.py
│   └── preprocessing.py
├── app.py
├── main.py
├── predict.py
├── requirements.txt
└── train.py

### Key Technical Features & Pipeline Design

#### 1. Custom IQR Outlier Clipping & Leakage Prevention
To ensure robust model generalization without data leakage, outlier boundaries are computed exclusively during the fitting phase on training split data (`src/preprocessing.py`).
* Calculates upper ($Q3 + 1.5 \times IQR$) and lower ($Q1 - 1.5 \times IQR$) bounds for numerical features.
* Persists bounds internally to apply consistent value clipping on evaluation and production inference samples.

#### 2. Class Imbalance Mitigation Strategy
The dataset exhibits significant target class imbalance (~87.6% negative vs. ~12.4% positive conversions).
* Evaluated performance using Precision, Recall, and ROC-AUC rather than standard Accuracy.
* Optimized XGBoost hyperparameters to penalize minority-class misclassifications effectively.

#### 3. Unified Pipeline Serialization
The feature processing transformers and the final XGBoost estimator are encapsulated into a single scikit-learn `Pipeline` object.
* Saved as a binary artifact (`models/xgboost_pipeline.joblib`) to mandate deterministic transformations across all deployment channels.
* Eliminates training-serving skew by keeping input transformation logic unified.

#### 4. Type Safety & Production Guardrails
Implemented dynamic data-type coercions prior to model evaluation.
* Ensures interface inputs (CLI / Streamlit) align with standard numerical data types before triggering transform steps.



### Dataset Analysis, Feature Engineering & Encoding Strategy

#### 1. Exploratory Data Analysis & Target Relationships
A thorough exploratory analysis was conducted across all 19 features to evaluate their correlation with customer conversion behavior:
* **High-Impact Behavioral Drivers**: Analysis revealed that engagement metrics—such as `TimeOnSite`, `PagesPerVisit`, and `ClickThroughRate` (CTR)—exhibited the strongest positive correlation with conversion outcomes.
* **Customer Demographics & History**: Features like `Income` and `LoyaltyPoints` demonstrated clear non-linear separation boundaries between converting and non-converting profiles.
* **Feature Redundancy & Selection**: Evaluated feature multicollinearity to prune low-variance or non-informative attributes, ensuring optimal model throughput without sacrificing accuracy.

#### 2. Feature Encoding & Data Transformation Strategy
To pass categorical and ordinal attributes into the estimator efficiently, transformations were customized per feature type:
* **One-Hot Encoding**: Applied to unordered nominal attributes (e.g., `CampaignChannel`, `CampaignType`, `AdvertisingPlatform`, `AdvertisingTool`) to prevent implicit numeric ranking assumptions.
* **Ordinal & Numerical Scaling**: Ordinal and numeric variables were standardized within the pipeline using standard scaling to align distance-based baselines and gradient descent trees.
* **Missing Value Imputation**: Automated median imputation strategy integrated directly inside custom transformers to prevent downstream missingness failures during inference.

---

### Quick Start & Model Usage

To evaluate or execute the pipeline, run the unified CLI controller from your terminal:

```bash
# Execute batch prediction and inference test
python main.py --mode predict

# Retrain the pipeline and export updated model artifacts
python main.py --mode train

# Launch the interactive Streamlit dashboard
python main.py --mode app