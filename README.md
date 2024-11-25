## Project Title: Customer Churn Prediction

### Overview
This project predicts customer churn using a machine learning approach. It utilizes the dataset **`WA_Fn-UseC_-Telco-Customer-Churn.csv`**, performing data analysis, preprocessing, and feature engineering before applying multiple machine learning models to evaluate predictive accuracy.

### Prerequisites
- Python 3.x
- Required Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install these libraries using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Dataset
The dataset includes customer data, with features like demographics, services subscribed, and monthly charges. The target variable, **`Churn`**, indicates whether a customer has discontinued the service.

### Steps in the Notebook

1. **Import Libraries**:
   Key Python libraries for data manipulation, visualization, and machine learning.

2. **Load Dataset**:
   Reads the dataset and displays its structure (`shape`, `info`, `head()`).

3. **Data Cleaning and Preprocessing**:
   - Check for null values and handle missing data.
   - Convert categorical variables to numerical using one-hot encoding.

4. **Exploratory Data Analysis (EDA)**:
   - Visualize numerical and categorical variables.
   - Analyze churn distribution across features.

5. **Feature Engineering**:
   - Identify numerical and categorical features.
   - Scale numerical features for consistent machine learning model inputs.

6. **Modeling**:
   Multiple models were trained and evaluated:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Decision Trees
   - K-Nearest Neighbors (KNN)

7. **Evaluation Metrics**:
   - Accuracy
   - Confusion Matrix
   - Classification Report

8. **Error Analysis**:
   - Plot error rates for KNN to determine optimal neighbors.

### Key Outputs
- Insights from EDA
- Performance comparison of models
- Optimal parameters for the KNN classifier
- Confusion matrices and accuracy scores for each model

### Usage
Run the notebook cell by cell to reproduce results. Ensure the dataset is in the working directory and named `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### Results
The project evaluates multiple models and identifies the most suitable one based on accuracy and interpretability.

### Acknowledgments
This project was inspired by real-world business challenges in predicting customer behavior to improve retention strategies.


