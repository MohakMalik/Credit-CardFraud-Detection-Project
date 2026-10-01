# Credit Card Fraud Detection

A machine learning project for detecting potentially fraudulent credit card transactions using **Random Forest Classifier** and **Streamlit**.

## Project Overview

Credit card fraud detection is a classification problem where the goal is to identify whether a transaction is legitimate or fraudulent.

This project covers:

* Data preprocessing and exploratory data analysis
* Feature engineering
* Categorical encoding
* Frequency encoding
* Feature scaling
* Machine learning model comparison
* Model evaluation using precision, recall, F1-score and PR-AUC
* Saving the final trained model
* Deploying the model using Streamlit

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

## Machine Learning Models

The following classification models were evaluated:

* Logistic Regression
* Gaussian Naive Bayes
* Decision Tree
* Random Forest
* XGBoost

Because the dataset is highly imbalanced, accuracy was not considered the only evaluation metric.

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* PR-AUC

## Final Model

**Random Forest Classifier** was selected as the final model based on the evaluation results.

It achieved:

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.999 |
| Precision | 0.970 |
| Recall    | 0.770 |
| F1-score  | 0.859 |
| PR-AUC    | 0.923 |

XGBoost achieved slightly higher recall, while Random Forest achieved higher precision, F1-score and PR-AUC on the test set.

Therefore, Random Forest was selected according to the evaluation criteria used in this project.

## Preprocessing

The following preprocessing steps were performed:

1. Removed unnecessary columns
2. Converted transaction date and time information
3. Calculated customer age
4. Extracted transaction hour
5. Encoded gender
6. Applied frequency encoding to:

   * Job
   * Merchant
7. Applied One-Hot Encoding to transaction category
8. Applied StandardScaler to numerical features
9. Maintained the same feature order used during model training

The fitted preprocessing objects were saved together with the model in:

```text
fraud_detection_model.pkl
```

The saved bundle contains:

```text
model
scaler
encoder
job_freq
merchant_freq
feature_columns
```

## Streamlit Application

The Streamlit application allows the user to enter transaction information and receive a fraud prediction.

The application displays:

* Legitimate Transaction
* Fraudulent Transaction
* Fraud Probability

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py
├── fraud_detection_model.pkl
├── requirements.txt
├── README.md
└── preprocessing.ipynb
```

## Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd Credit-Card-Fraud-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

## Model File

The application requires:

```text
fraud_detection_model.pkl
```

This file contains the trained Random Forest model along with the preprocessing objects required for prediction.

Do not retrain the model when running the Streamlit application.

## Learning Note

This project was developed as part of my machine learning learning and practice work.

The project focuses on understanding:

* Classification
* Imbalanced datasets
* Feature engineering
* Categorical encoding
* Feature scaling
* Model evaluation
* Precision-recall trade-offs
* PR-AUC
* Model serialization
* Machine learning deployment with Streamlit

The implementation may include techniques learned through courses, tutorials, documentation, and experimentation. The purpose is to understand and apply the concepts in a practical project.

## Future Improvements

Possible improvements include:

* Hyperparameter optimization
* Better handling of class imbalance
* Threshold optimization
* Cross-validation
* More detailed fraud analysis
* Model explainability using SHAP
* Deployment to a cloud platform
* Monitoring model performance on new transactions

## Author

Mohak Malik

B.Tech Computer Science and Engineering

ABES Engineering College
