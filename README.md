# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using **Logistic Regression** and **Random Forest**, built in a Jupyter notebook.

## Dataset

The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group).

- 284,807 transactions made by European cardholders over two days
- 492 frauds (about 0.17%), so the data is **highly imbalanced**
- 30 features: `Time`, `Amount`, and `V1`-`V28` (anonymized PCA components)
- Target: `Class` (1 = fraud, 0 = legitimate)

> **The dataset is not included in this repository** because the file (~144 MB) exceeds GitHub's 100 MB limit. See the setup steps below to download it.

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
   cd YOUR-REPO
   ```

2. **Install dependencies** (a virtual environment is recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Go to the [Kaggle dataset page](https://www.kaggle.com/mlg-ulb/creditcardfraud) and download `creditcard.csv` (a free Kaggle account is required).
   - Place `creditcard.csv` in the same folder as the notebook.

4. **Run the notebook**
   ```bash
   jupyter notebook Credit_Card_Fraud_Detection.ipynb
   ```

## Workflow

1. Load the data and check its shape and missing values
2. Explore the class imbalance and feature correlations
3. Split into train/test sets (75% / 25%)
4. Scale features with `StandardScaler`
5. Train and evaluate two models

## Results

Evaluated on the 25% test set (71,202 transactions, 120 of them fraud):

| Model | Fraud Precision | Fraud Recall | Fraud F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.88 | 0.66 | 0.75 | 99.93% |
| Random Forest | 0.94 | 0.78 | 0.85 | 99.95% |

Because fewer than 0.2% of transactions are fraud, accuracy is misleading here (predicting "not fraud" every time would score about 99.8%). Precision, recall and F1 on the fraud class are the metrics that matter. Random Forest performs better on all of them.

## Possible Improvements

- Handle class imbalance (SMOTE, class weights, undersampling)
- Try gradient boosting models (XGBoost, LightGBM)
- Tune hyperparameters with cross-validation
- Evaluate with a precision-recall curve / AUPRC

## Tech Stack

Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter
