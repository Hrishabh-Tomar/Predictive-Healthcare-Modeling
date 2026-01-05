# Predictive-Healthcare-Modeling

Project Title: Healthcare Analytics Dashboard: Predicting Length of Stay & Billing

Project Overview This project applies predictive analytics to the healthcare sector, specifically forecasting Hospital Length of Stay (LOS) and Billing Amounts based on patient demographics and medical history. By integrating machine learning pipelines with an interactive web dashboard, this tool aims to assist healthcare administrators in optimizing resource allocation and operational planning.

Key Objectives

Data Integration: Ingest, clean, and merge raw healthcare data for comprehensive analysis.

Preprocessing: Implement robust data cleaning (casing correction, typo handling) and datetime conversion for accurate time-series analysis.

Feature Engineering: Derive critical metrics such as 'Length of Stay' and standardized 'Age Groups'.

Predictive Modeling: Develop Linear Regression models to estimate costs and duration of hospitalization.

Deployment: Launch a real-time, interactive Dash application for end-user accessibility.

Technology Stack

Data Processing: Pandas, NumPy

Modeling: Scikit-learn (Pipelines, OneHotEncoder, LinearRegression)

Visualization: Matplotlib, Seaborn, Plotly

Deployment: Dash by Plotly (Bootstrap Components)

Methodology

Data Wrangling: ADDressed data quality issues, including standardizing inconsistent string casing and resolving specific entry errors (e.g., "United Healthcare" typos).

Exploratory Data Analysis (EDA): Conducted univariate and bivariate analysis to identify correlations between age, medical conditions, and billing trends using scatter plots and correlation matrices.

Model Architecture: Utilized Scikit-learn pipelines to streamline preprocessing (One-Hot Encoding) and model fitting.

Evaluation: Models were assessed using Mean Absolute Error (MAE) to quantify prediction accuracy.

Results & Dashboard The analysis culminated in a Dash web application where users can input patient details (Age, Condition, Admission Type) to receive instant predictions.

Billing Model Performance: MAE of ~$12,189.

Length of Stay Performance: MAE of ~7.5 days.

Conclusion This project demonstrates how linear regression models can effectively translate raw hospital data into actionable operational insights. Future iterations could improve accuracy by incorporating non-linear models (e.g., Random Forest) or expanding the feature set.
