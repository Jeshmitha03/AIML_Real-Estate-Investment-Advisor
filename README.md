# AIML_Real-Estate-Investment-Advisor
🏠 ML-powered Real Estate Investment Advisor that predicts property investment potential and estimates 5-year future value using Python, Scikit-learn, MLflow, and Streamlit.

An end-to-end Machine Learning project that analyzes real-estate properties, predicts **investment potential**, and estimates the **5-year future property value** using Python, Scikit-learn, MLflow, and Streamlit.

---

## 📌 Project Overview

The **Real Estate Investment Advisor** is a Machine Learning application designed to help users evaluate properties based on factors such as:

* Property price
* Property size
* Location
* BHK
* Property type
* Property age
* Amenities
* Nearby schools
* Nearby hospitals
* Public transportation
* Furnishing status
* Floor information
* Other property characteristics

The project applies **Data Cleaning, Feature Engineering, Exploratory Data Analysis, Machine Learning, Hyperparameter Tuning, Experiment Tracking, and Streamlit Deployment** to build an end-to-end ML solution.

---

## 🎯 Project Objectives

### 1. Classification

Predict whether a property is a:

* ✅ **Good Investment**
* ❌ **Not a Good Investment**

The classification problem is solved using supervised machine learning algorithms.

### 2. Regression

Estimate the property's:

**📈 Future Property Value After 5 Years**

The regression model predicts an estimated future property price based on the available property features.

> ⚠️ The 5-year value is a project-based estimate and should not be considered a guaranteed market forecast.

---

## 🔄 Project Workflow

```text
Data Collection
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
Data Preprocessing
       ↓
Train-Test Split
       ↓
Classification Models
       ↓
Regression Models
       ↓
Model Evaluation
       ↓
Hyperparameter Tuning
       ↓
MLflow Experiment Tracking
       ↓
Best Model Selection
       ↓
Model Saving
       ↓
Streamlit Dashboard
```

---

## 🧹 Data Preprocessing

The following preprocessing techniques were applied:

* Missing value handling
* Duplicate detection and removal
* Data type conversion
* Numerical feature cleaning
* Categorical feature handling
* Median imputation for numerical values
* Mode/`Unknown` handling for categorical values
* Feature scaling
* One-hot encoding

---

## 🛠️ Feature Engineering

Several new features were created to improve the analysis and model performance.

### Calculated Price per SqFt

```text
Calculated_Price_per_SqFt =
(Price_in_Lakhs × 100000) / Size_in_SqFt
```

### Property Age

```text
Calculated_Age_of_Property =
2026 - Year_Built
```

### Amenities Count

Counts the number of amenities available for a property.

### Amenity Density Score

Measures the density of amenities relative to property size.

### Floor Ratio

```text
Floor_Ratio =
Floor_No / Total_Floors
```

### Price Range

Properties were categorized into different price ranges:

* Below 50L
* 50L–1Cr
* 1Cr–2Cr
* 2Cr–5Cr
* Above 5Cr

### Size Category

Properties were grouped into:

* Small
* Medium
* Large
* Very Large
* Luxury

### BHK Category

Properties were categorized as:

* 1 BHK
* 2 BHK
* 3 BHK
* 4 BHK
* 5+ BHK

---

## 📊 Exploratory Data Analysis

EDA was performed to understand relationships between property characteristics and prices.

Key analyses included:

* Price distribution
* Property size distribution
* Price per square foot
* Size vs property price
* Price outlier detection
* Average price by state
* Average price by city
* Property age by locality
* BHK distribution
* Price analysis of expensive localities
* Correlation analysis
* Schools vs property price
* Hospitals vs property price
* Furnishing status vs price
* Facing vs price per square foot
* Owner type distribution
* Availability status
* Parking analysis
* Amenities vs price
* Public transport vs investment potential

---

# 🤖 Machine Learning

## Classification Models

The following classification algorithms were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Gradient Boosting

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

The model with the best F1-score was selected for the final classification pipeline.

---

## 📈 Regression Models

The following regression algorithms were evaluated:

1. Linear Regression
2. Ridge Regression
3. Decision Tree Regression
4. Random Forest Regression
5. Gradient Boosting Regression

### Evaluation Metrics

* MAE
* RMSE
* R² Score

The model with the lowest RMSE was selected for the final regression pipeline.

---

# ⚙️ Hyperparameter Tuning

Hyperparameter optimization was performed using:

* `GridSearchCV`
* 5-Fold Cross Validation
* `StratifiedKFold` for classification
* `KFold` for regression

This process was used to identify better-performing model configurations.

---

# 📦 ML Pipeline

The project uses Scikit-learn pipelines to combine:

```text
Raw Input
    ↓
Data Preprocessing
    ↓
Numerical Scaling
    ↓
Categorical Encoding
    ↓
Machine Learning Model
    ↓
Prediction
```

Using a complete pipeline ensures that the same preprocessing steps are applied during both training and prediction.

---

# 📊 MLflow Experiment Tracking

**MLflow** was used to track machine learning experiments.

The following information was logged:

* Model parameters
* Evaluation metrics
* Model artifacts
* Classification experiments
* Regression experiments
* Tuned model results

This makes it easier to compare experiments and reproduce model results.

---

# 💾 Model Saving

The trained models were saved using **Joblib**.

```text
models/
│
├── best_classification_model.pkl
├── best_regression_model.pkl
├── model_info.pkl
└── feature_columns.pkl
```

The saved classification and regression models contain the complete preprocessing and prediction pipeline.

---

# 🚀 Streamlit Dashboard

The project includes an interactive **Streamlit dashboard**.

The dashboard allows users to enter property information and receive predictions.

### Dashboard Features

#### 🏠 Prediction

Users can enter:

* State
* City
* Locality
* Property Type
* BHK
* Property Size
* Property Price
* Year Built
* Furnished Status
* Floor Number
* Total Floors
* Nearby Schools
* Nearby Hospitals
* Public Transport Accessibility
* Parking
* Security
* Amenities
* Facing
* Owner Type
* Availability Status

The application then provides:

* Investment prediction
* Investment probability
* Estimated future property value
* Estimated appreciation

---

## 📊 Market Insights

The dashboard provides:

* Total properties
* Average property price
* Average property size
* Average price per square foot
* Price distribution
* Top cities
* Property type distribution
* BHK distribution

---

## 💰 Price Analysis

The dashboard provides:

* Average price by property type
* Size vs price analysis
* Price per square foot distribution
* Price range analysis
* Average price by state

---

# 🗂️ Project Structure

```text
GUVI_3_project/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── india_housing_prices.csv
│   │
│   └── processed/
│       ├── cleaned_housing_prices.csv
│       ├── feature_engineered_housing_prices.csv
│       └── ml_ready_housing_prices.csv
│
├── models/
│   ├── best_classification_model.pkl
│   ├── best_regression_model.pkl
│   ├── model_info.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── Real_Estate_Investment_Advisor.ipynb
│
├── venv/
│
├── requirements.txt
│
└── README.md
```

---

# 🧰 Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Experiment Tracking

* MLflow

### Model Serialization

* Joblib

### Web Application

* Streamlit

### Development Environment

* VS Code
* Jupyter Notebook

---

# 📥 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Navigate to the project directory:

```bash
cd GUVI_3_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

From the project root:

```bash
streamlit run dashboard/app.py
```

The application will open in your browser.

---

# 📈 Machine Learning Architecture

```text
                    Real Estate Dataset
                            │
                            ↓
                    Data Preprocessing
                            │
                            ↓
                    Feature Engineering
                            │
                            ↓
                    ┌───────┴────────┐
                    ↓                ↓
             Classification       Regression
                    │                │
                    ↓                ↓
             Investment        Future Property
               Potential            Value
                    │                │
                    ↓                ↓
              Model Tuning       Model Tuning
                    │                │
                    └───────┬────────┘
                            ↓
                       MLflow Tracking
                            │
                            ↓
                       Best Models
                            │
                            ↓
                    Streamlit Dashboard
```

---

# ⚠️ Important Project Note

The `Good_Investment` target is created using **project-defined/domain rules**, rather than an observed real-world investment outcome.

Similarly, the 5-year property value is a **project-based estimated projection**.

Therefore, the model outputs should be interpreted as analytical estimates rather than guaranteed investment recommendations or actual future market prices.

For a production-grade system, the targets should ideally be trained using historical future property prices and real investment outcomes.

---

# 🔮 Future Improvements

Possible improvements include:

* Use real historical property price data
* Add actual property appreciation data
* Incorporate location-based geospatial features
* Add distance to metro stations and major roads
* Add economic indicators
* Add interest-rate information
* Add real-time property listings
* Try XGBoost and LightGBM
* Implement SHAP explainability
* Add model monitoring
* Deploy the application to Streamlit Cloud
* Add user authentication
* Build an API using FastAPI

---

# 👨‍💻 Skills Demonstrated

This project demonstrates practical knowledge of:

* Python
* Pandas
* NumPy
* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis
* Data Visualization
* Supervised Machine Learning
* Classification
* Regression
* Model Evaluation
* Cross Validation
* Hyperparameter Tuning
* Scikit-learn Pipelines
* MLflow
* Model Serialization
* Streamlit
* End-to-End ML Project Development

---

## ⭐ Conclusion

The **Real Estate Investment Advisor** demonstrates an end-to-end Machine Learning workflow, starting from raw real-estate data and ending with an interactive Streamlit application.

The project combines **data preprocessing, feature engineering, EDA, supervised learning, hyperparameter tuning, MLflow experiment tracking, model deployment, and interactive visualization** into a complete ML application.
