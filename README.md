# customer-churn-prediction

## 📊 Project Overview
This project tackles the critical business problem of **customer churn** for a telecommunications company. Customer churn, the rate at which customers stop doing business with a company, is a key metric and a primary focus for customer retention strategies. This project performs a comprehensive analysis of customer data and builds a machine learning model to predict whether a customer is likely to churn.

By identifying at-risk customers, the business can proactively implement targeted retention campaigns, thereby reducing churn and increasing profitability.

## 🎯 Objectives
- **Analyze** customer data to uncover key factors influencing churn.
- **Preprocess** the dataset by handling missing values and encoding categorical variables.
- **Build and Train** a Random Forest Classifier to predict customer churn.
- **Evaluate** the model's performance using accuracy and a confusion matrix.
- **Provide actionable insights** that can help in formulating customer retention strategies.

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - `pandas`, `numpy` (Data Manipulation)
  - `matplotlib`, `seaborn` (Data Visualization)
  - `scikit-learn` (Machine Learning: Preprocessing, Model Training, Evaluation)

## 📁 Dataset
- **Source:** The project uses the publicly available `Telco-Customer-Churn.csv` dataset.
- **Description:** The dataset contains information about 7,043 telecom customers, including:
  - **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `StreamingTV`, etc.
  - **Account Information:** `tenure` (months with company), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - **Target Variable:** `Churn` (Yes/No)

## 🔬 Methodology & Steps

### 1. Data Loading & Initial Exploration
- Loaded the dataset and inspected the first few rows to understand its structure.
- Checked for missing values and data types.
- Generated descriptive statistics for numerical features.

### 2. Data Preprocessing & Cleaning
- **Handled Missing Values:** The `TotalCharges` column had missing values represented as empty strings. These were converted to numerical type and filled with the column's median.
- **Encoded Categorical Variables:** All categorical columns (e.g., `gender`, `Partner`, `Churn`, etc.) were converted into numerical format using `LabelEncoder` to make them suitable for the machine learning model.
- **Dropped Irrelevant Features:** The `customerID` column was removed as it is a unique identifier and not useful for prediction.

### 3. Exploratory Data Analysis (EDA)
- Visualized the distribution of the target variable (`Churn`) using a count plot to understand the class imbalance.
- *Further EDA (like analyzing churn rates by `Contract` type, `tenure`, etc.) can be added to gain deeper insights.*

### 4. Feature Engineering & Model Preparation
- **Split Data:** The data was split into features (`X`) and the target variable (`y`).
- **Train-Test Split:** The dataset was divided into a training set (80%) and a testing set (20%) to evaluate the model's performance on unseen data.
- **Feature Scaling:** Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using `StandardScaler` to ensure they were on a similar scale, which improves the performance of many ML algorithms.

### 5. Model Building & Training
- **Algorithm:** A **Random Forest Classifier** was chosen for this task due to its high accuracy, robustness to overfitting, and ability to handle complex relationships within the data.
- The model was trained on the preprocessed training data (`X_train`, `y_train`).

### 6. Model Evaluation
- **Predictions:** The trained model was used to make predictions on the test set (`X_test`).
- **Accuracy:** The model achieved an **accuracy of 78%** on the test set.
- **Confusion Matrix:** A confusion matrix was plotted to visualize the model's performance in terms of True Positives, True Negatives, False Positives, and False Negatives.

## 📈 Results
- The final Random Forest model successfully predicted customer churn with **78% accuracy**.
- The confusion matrix provides a detailed breakdown of the prediction outcomes, which is crucial for understanding the trade-off between correctly identifying churners and avoiding false alarms.

## 🚀 How to Run
1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/customer-churn-prediction.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd customer-churn-prediction
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure the `Telco-Customer-Churn.csv` file is in the same directory as the notebook.
5.  Run the Jupyter Notebook `Customer_Churn_Analysis_Prediction.ipynb` cell by cell.

## 📝 Future Work
- Address the class imbalance in the target variable using techniques like SMOTE.
- Perform hyperparameter tuning (e.g., using GridSearchCV) to optimize the model's performance.
- Experiment with other classification algorithms (e.g., XGBoost, Logistic Regression).
- Deploy the model as a web application using Flask or Streamlit for real-time predictions.

