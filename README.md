# Predictive-Healthcare-Modeling

Healthcare Data Analysis & Prediction Dashboard
📌 Project Overview
This project provides an end-to-end analysis of a healthcare dataset, focusing on patient demographics, billing patterns, and hospital stay durations. It moves from data cleaning and Exploratory Data Analysis (EDA) to Machine Learning, culminating in an interactive Dash web application that predicts Length of Stay and Billing Amount based on patient criteria.

🚀 Key Features
1. Data Wrangling & Cleaning
The raw data is processed to ensure consistency:

Normalization: Converts names, hospitals, and insurance providers to Title Case.

Correction: Fixes specific typos (e.g., standardizing "unitedhealthcare" to "United Healthcare").

Formatting: Converts Date of Admission and Discharge Date to datetime objects for time-series analysis.

2. Exploratory Data Analysis (EDA)
The project visualizes key trends using Matplotlib and Seaborn:

Demographics: Frequency distributions of Gender, Blood Type, and Medical Conditions.

Financials: Scatter plots and box plots analyzing Billing Amounts against Age and Gender.

Time Series: Yearly trends for Admissions, Discharges, and total Billing Amounts.

Hospital Operations: Analysis of Average Length of Stay by Medical Condition.

3. Machine Learning Pipelines
Two Linear Regression models are implemented using scikit-learn pipelines to predict hospital metrics.

Model 1: Billing Amount Prediction

Features: Age Group, Medical Condition

Preprocessor: One-Hot Encoding for categorical variables.

Model 2: Length of Stay Prediction

Target: Calculated as Discharge Date - Date of Admission.

Features: Age, Medical Condition, Admission Type.

Insights: The model identifies specific factors increasing stay duration (e.g., specific ages or conditions like Asthma) vs. those decreasing it (e.g., Urgent admissions).

4. Interactive Dashboard
A web-based interface built with Dash and Bootstrap allows users to input patient details and receive real-time predictions.

🛠️ Tech Stack
Language: Python

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

Machine Learning: Scikit-Learn

Web Framework: Dash, Dash Bootstrap Components

⚙️ Installation & Usage
Clone the Repository

Bash

git clone https://github.com/your-username/healthcare-analysis.git
cd healthcare-analysis
Install Dependencies

Bash

pip install pandas numpy matplotlib seaborn plotly scikit-learn dash dash-bootstrap-components
Setup Dataset

Ensure healthcare_dataset.csv is on your machine.

Update the file path in the script to match your local directory:

Python

# In healthcare-dataset-new-first-proj-1.py
df = pd.read_csv(r'path\to\your\healthcare_dataset.csv')
Run the Application Execute the script to train the models and launch the server:

Bash

python healthcare-dataset-new-first-proj-1.py
Access the Dashboard Open your web browser and navigate to: http://127.0.0.1:8050/

📊 Feature Engineering
To improve model performance, the following features were engineered:

Length of Stay: Derived from the difference between discharge and admission dates.

Age Group: Patients were categorized into groups (Child, Young Adult, Adult, Middle Aged, Senior) to capture non-linear age trends.

📉 Model Performance
The models are evaluated using Mean Absolute Error (MAE) to determine the average deviation between predicted and actual values. Feature importance charts are generated to visualize which factors most significantly impact billing and length of stay.

📝 License
This project is open-source.
