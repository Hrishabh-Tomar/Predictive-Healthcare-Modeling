# Predictive-Healthcare-Modeling

Healthcare Data Analysis & Prediction Dashboard
📌 Project Overview
This project provides an end-to-end analysis of a healthcare dataset, focusing on patient demographics, billing patterns, and hospital stay durations. It moves from data cleaning and Exploratory Data Analysis (EDA) to Machine Learning, culminating in an interactive Dash web application that predicts Length of Stay and Billing Amount based on patient criteria.

Data Preprocessing
Before modeling, the raw data undergoes several cleaning and formatting steps:

Data Loading: The dataset is loaded using Pandas.

Date Conversion: Date of Admission and Discharge Date columns are converted to datetime objects to facilitate time-based calculations.

Categorical Handling: Inconsistent casing in text columns (e.g., 'Gender', 'Hospital') and specific typos (e.g., correcting "unitedhealthcare") are addressed to ensure uniformity.

Feature Engineering
New features were derived from the existing data to improve model accuracy:

Length of Stay: Calculated as the difference in days between the Discharge Date and the Date of Admission.

Python

df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
Age Group: Patients were categorized into age groups to capture broader trends in billing and recovery time.

Child: < 18

Young Adult: 18–29

Adult: 30–49

Middle Aged: 50–64

Senior: 65+

Model Creation
Two separate Linear Regression models were created using scikit-learn pipelines. These pipelines ensure that preprocessing steps like encoding are automatically applied to new data.

1. Billing Amount Prediction Model
Target Variable: Billing Amount

Features Used:

Age Group (Categorical)

Medical Condition (Categorical)

Pipeline Steps:

One-Hot Encoding: Converts categorical variables (Age Group, Medical Condition) into numeric binary vectors.

Linear Regression: Fits a linear equation to predict the continuous billing amount.

2. Length of Stay Prediction Model
Target Variable: Length of Stay

Features Used:

Age (Numerical)

Medical Condition (Categorical)

Admission Type (Categorical: e.g., Emergency, Elective)

Pipeline Steps:

One-Hot Encoding: Applied to Medical Condition and Admission Type.

Linear Regression: Fits a linear equation to predict the number of days a patient will stay.

Model Evaluation
The models are evaluated using Mean Absolute Error (MAE), which measures the average magnitude of errors in the predictions.

The data is split into training (80%) and testing (20%) sets using train_test_split to ensure the model is tested on unseen data.

Interactive Dashboard
The project includes a Dash web application (dash, dash_bootstrap_components) that allows users to input patient details and receive real-time predictions.

Inputs: User selects Age, Medical Condition, Admission Type, and Age Group.

Outputs: The app displays the predicted "Length of Stay" (in days) and "Billing Amount" (in USD).

Dependencies
To run this project, the following libraries are required:

pandas

numpy

scikit-learn

matplotlib

seaborn

plotly

dash

dash-bootstrap-components
