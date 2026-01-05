#!/usr/bin/env python
# coding: utf-8

# ####IMPORT LIBRARIES

# In[1]:


import pandas as pd
import numpy as pd
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import plotly


# Ignore warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# In[2]:


# Define the correct filepath by escaping backslashes
filepath =  r"C:\Users\ghode\OneDrive - NHS\Desktop\healthcare_dataset.csv"


# Read the CSV file into a DataFrame
df = pd.read_csv(filepath)
print (df)


# In[3]:


df.info()


# In[4]:


df.columns


# In[5]:


df.head(15)


# DATA WRANGLING
# 

# In[6]:


def wrangle(filepath):
    #read csv into filepath
    df = pd.read_csv(filepath)
    #correct casing for name column
    df['Name'] = df['Name'].str.title()
    # correct casing for df columns
    df[['Gender', 'Hospital', 'Doctor', 'Insurance Provider']] = df[['Gender', 'Hospital', 'Doctor', 'Insurance Provider']].apply(lambda x: x.str.title())
    # correct spacing in col insurance provider
    df['Insurance Provider'] = df['Insurance Provider'].str.strip()
    #correct spacing in a stringin column insurance provider
    df['Insurance Provider'] = df['Insurance Provider'].str.replace("unitedhealthcare", "United Healthcare", case=False)
    #drop null and na values
    df.dropna
    #check for duplicates in rows
    df.duplicated()
    # Check for duplicate values in columns
    duplicate_values = df[["Name", "Age", "Gender", "Blood Type", "Medical Condition",
       "Date of Admission", "Doctor", "Hospital", "Insurance Provider",
       "Billing Amount", "Room Number", "Admission Type", "Discharge Date",
       "Medication", "Test Results"]].duplicated()
    return df


# In[7]:


df.head(10)


# EXPLORATORY DATA ANALYSIS
# 
# 

# In[8]:


#descriptive analysis
summary_stats_numerical=df.describe(include ="number")
print(summary_stats_numerical)


# In[9]:


#Scatter plot Age Vs Billing amount 
#Rounding up billing amount into 2 decimal points for better visualization
df[["Billing Amount"]]
df["Billing Amount"] = df[["Billing Amount"]].round(2)
print (df[["Billing Amount"]])


# In[10]:


#Scatter plot Age Vs Billing amount 
# Create scatter plot
plt.figure(figsize=(8, 6))
# Plot age vs. billing amount
plt.scatter(df["Age"], df["Billing Amount"], alpha=0.5) 
#title
plt.title("Scatter Plot of Age vs. Billing Amount")  
 # Label x-axis and y-axis
plt.xlabel("Age")  
plt.ylabel("Billing Amount")  
plt.show();


# In[11]:


#Since data is overlapping, it shows data is too much to get a good scatter plot so ill use the mean
# Calculate mean billing amount for each age
mean_billing_by_age = df.groupby("Age")["Billing Amount"].mean()

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(mean_billing_by_age.index, mean_billing_by_age.values, alpha=0.5) 

# Title
plt.title("Scatter Plot of Mean Billing Amount by Age")  

# Label x-axis and y-axis
plt.xlabel("Age")  
plt.ylabel("Mean Billing Amount")  

plt.show();


# In[12]:


# Calculate correlation coefficient
correlation = df["Age"].corr(df["Billing Amount"])

print("Correlation coefficient between age and billing amount:", correlation)


# In[13]:


#create col mean billing amount
mean_billing_amount = df["Billing Amount"].mean()

# Create a new column with the mean billing amount for each row
df["Mean_billing_amount"] = mean_billing_amount

print(df)


# In[14]:


# Calculate correlation coefficient
correlation = df["Age"].corr(df["Mean_billing_amount"])

print("Correlation coefficient between age and Mean_billing_amount:", correlation)


# In[15]:


df.rename(columns={"Mean_billing_amount': 'Mean billing amount"}, inplace=True)
print(df)


# In[16]:


##plotting billing amount vs gender on a pie chart, vertical bar chart and box & whisker plot
#pie chart Gender vs billing amount on a pie chart
# Creating dataset
total_billing_by_gender = df.groupby("Gender")["Billing Amount"].sum()
# Create pie chart
plt.figure(figsize=(8, 6))
plt.pie(total_billing_by_gender, 
        labels=total_billing_by_gender.index, 
        autopct="%1.1f%%", 
        colors=["blue", "pink"])
plt.title("Total Billing Amount by Gender")
#Equal aspect ratio ensures that pie is drawn as a circle
plt.axis("equal")  
plt.show();


# In[17]:


# create vertical bar chart
average_billing_by_gender = df.groupby("Gender")["Billing Amount"].mean()
# Create vertical bar chart
plt.figure(figsize=(8, 6))
average_billing_by_gender.plot(kind="bar", color=["blue", "pink"])
plt.title("Average Billing Amount by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Billing Amount")
#Rotate x-axis labels 
plt.xticks(rotation=0) 
plt.show()


# In[18]:


# Create box plot
plt.figure(figsize=(8, 6))
df.boxplot(column="Billing Amount", by="Gender", grid=False)
plt.title("Box Plot of Billing Amount by Gender")
plt.xlabel("Gender")
plt.ylabel("Billing Amount")
 # Optional: Set x-axis labels
plt.xticks([1, 2], ["Male", "Female"])  
plt.show();


# frequency distrubution of gender, blood type and medical condition

# In[19]:


#frequency distrubution
gender_counts=df["Gender"].value_counts()
print(df["Gender"])
print("Frequency distribution of gender:\n", gender_counts)


# In[20]:


blood_counts=df["Blood Type"].value_counts()
print(df["Blood Type"])
print("Frequency distribution of Blood Type:\n", blood_counts)


# In[21]:


conditions_counts=df["Medical Condition"].value_counts()
print(df["Medical Condition"])
print("Frequency distribution of Medical Condition:\n", conditions_counts)


# In[22]:


# Visualize the frequency distribution using a bar plot
plt.figure(figsize=(8, 6))
gender_counts.plot(kind='bar', color='skyblue')
plt.title("Frequency Distribution of Gender")
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  
plt.show();


# In[23]:


# Visualize the frequency distribution using a bar plot
plt.figure(figsize=(8, 6))
blood_counts = df["Blood Type"].value_counts()

#color palette for each blood type
blood_colors = {'A+': 'red', 'A-': 'blue', 'B+': 'green', 'B-': 'orange', 'AB+': 'purple', 'AB-': 'pink', 'O+': 'yellow', 'O-': 'cyan'}

# Visualize the frequency distribution using a bar plot
plt.figure(figsize=(8, 6))
blood_counts.plot(kind='bar', color=[blood_colors[blood_type] for blood_type in blood_counts.index])  
plt.title('Frequency Distribution of Blood Type')
plt.xlabel('Blood Type')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  
plt.show();


# In[24]:


# Visualize the frequency distribution using a bar plot
plt.figure(figsize=(8, 6))
conditions_counts.plot(kind="bar", color="violet")
plt.title("Frequency Distribution of Medical Conditions")
plt.xlabel("Medical Conditons")
plt.ylabel("Frequency")
plt.xticks(rotation=0)  
plt.show();


# In[25]:


#creating a plot of medical condition vs gender
#crosstabulation of gender & med condition
gender_med_condition_counts=pd.crosstab(df["Gender"],df["Medical Condition"])
#create plot
gender_med_condition_counts.plot(kind="bar", figsize=(12,8) )
plt.title("Medical Conditions by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Medical Conditions")
plt.xticks(rotation=0)  
plt.show();


# In[26]:


#creating a plot of gender vs bloodtype
#crosstabulation of gender & bloodtype
gender_blood_counts=pd.crosstab(df["Gender"],df["Blood Type"])
#create plot
gender_blood_counts.plot(kind="bar", figsize=(12,8) )
plt.title("Blood Types according to Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Blood Types")
plt.xticks(rotation=0)  
plt.show();


# In[27]:


#creating a plot of gender vs insurance
#crosstabulation of gender & insurance
insurance_gender_counts=pd.crosstab(df["Gender"],df["Insurance Provider"])
#create plot
insurance_gender_counts.plot(kind="bar", figsize=(12,8))
plt.title("Insurance Companies usage according to Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Insurance Provider")
plt.xticks(rotation=0)  
plt.show();


# In[28]:


# Group the data by medical conditions and calculate mean billing amount
mean_billing_by_condition = df.groupby('Medical Condition')['Billing Amount'].mean()

# Create a bar plot
plt.figure(figsize=(12, 8))
mean_billing_by_condition.plot(kind='bar', color='skyblue')
plt.title('Mean Billing Amount by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Mean Billing Amount')
plt.xticks(rotation=0)
plt.show();


# In[29]:


#frequency distrubution
admission_counts=df["Date of Admission"].value_counts()
print(df["Date of Admission"])
print("Frequency distribution of Date of Admissions:\n", admission_counts)


# In[30]:


admissions_by_gender=df.groupby("Gender")["Date of Admission"].count()
print(admissions_by_gender)


# In[31]:


#create bar chart
admissions_by_gender = df.groupby('Gender')['Date of Admission'].count()

# Plot the data
plt.figure(figsize=(8, 6))
admissions_by_gender.plot(kind="bar", color=["pink", "blue"])
plt.title("Number of Admissions by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Admissions")
plt.xticks(rotation=0)
plt.show();


# In[32]:


# Convert to datetime format 
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])


# In[33]:


#length of stay vs Gender
#Calculate the length of stay in days
#create col length of stay
df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

# Display the DataFrame to check the new column
print(df[["Date of Admission", "Discharge Date", "Length of Stay"]])


# In[34]:


#  summary statistics for Length of Stay
length_of_stay_stats = df.groupby('Gender')['Length of Stay'].describe()
print(length_of_stay_stats)


# In[35]:


print(df.columns)

# Calculate the average length of stay by medical condition
average_length_of_stay_by_condition = df.groupby("Medical Condition")["Length of Stay"].mean()

# Plot the bar chart
plt.figure(figsize=(12, 8))
average_length_of_stay_by_condition.plot(kind='bar', color='skyblue')
plt.title("Average Length of Stay by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel('Average Length of Stay (days)')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels 
plt.grid(axis='y')
plt.show();


# In[36]:


#create boxplot so we can dentify any outliers in the length of stay by gender.
plt.figure(figsize=(10, 6))
df.boxplot(column='Length of Stay', by='Gender', grid=False)
plt.title('Length of Stay by Gender')
# Suppress the automatic title
plt.suptitle('') 
plt.xlabel('Gender')
plt.ylabel('Length of Stay (days)')
plt.show();


# In[37]:


# Calculate the average length of stay by medical condition
average_length_of_condition = df.groupby("Medical Condition")["Length of Stay"].mean()

# Plot the bar chart
plt.figure(figsize=(10, 6))
average_length_of_condition.plot(kind='bar', color=['blue', 'pink'])
plt.title("Average Length of Stay by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel('Average Length of Stay (days)')
plt.xticks(rotation=0)
plt.show();


# In[38]:


print(df)


# In[39]:


# Display the df for length of stay vs billing amount
print(df[["Length of Stay","Billing Amount"]])


# In[40]:


#plot length of stay vs billing amount
#scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df["Length of Stay"], df[  "Mean billing amount"], alpha=0.5)
plt.title("Length of Stay vs. Mean billing amount")
plt.xlabel("Length of Stay (days)")
plt.ylabel("Mean billing Amount")
plt.show();


# In[41]:


# Calculate correlation coefficient for length of stay vs billing amt
correlation = df["Length of Stay"].corr(df['Billing Amount'])

print("Correlation coefficient between Length of Stay and billing amount:", correlation)


# In[42]:


 #Calculate correlation coefficient for length of stay vs mean billing amt
correlation = df["Length of Stay"].corr(df['Mean billing amount'])

print("Correlation coefficient between Length of Stay and Mean billing amount:", correlation)


# In[43]:


# Display the df for length of stay vs age
print(df[[ 'Length of Stay',  'Age']])
#plot length of stay vs age
#scatter plot
# Group the DataFrame by age and calculate the mean length of stay for each age group
mean_length_of_stay_by_age = df.groupby('Age')['Length of Stay'].mean().reset_index()
# Plot mean length of stay vs. age
plt.figure(figsize=(10, 6))
plt.plot(mean_length_of_stay_by_age['Age'], mean_length_of_stay_by_age['Length of Stay'], marker='o')
plt.title("Mean Length of Stay vs. Age")
plt.xlabel("Age")
plt.ylabel("Mean Length of Stay (days)")
plt.grid(True)
plt.show();



# In[44]:


# Calculate correlation coefficient for length of stay vs Age
correlation = df["Length of Stay"].corr(df["Age"])

print("Correlation coefficient between Length of Stay and age:", correlation)


# In[46]:


# Plot histogram of ages
plt.figure(figsize=(8, 6))
plt.hist(df["Age"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Ages")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show();


# In[45]:


df.head(10)


# In[47]:


# Create a box plot for the length of stay
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['Length of Stay'], color='skyblue')
plt.title('Box Plot of Length of Stay')
plt.ylabel('Length of Stay (days)')
plt.show();


# TIME SERIES DATA VISUALIZATION

# In[48]:


# Ensure the 'Date of Admission' and 'Discharge Date' columns are in datetime format
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# Extract the year for grouping
df['Admission Year'] = df['Date of Admission'].dt.year
df['Discharge Year'] = df['Discharge Date'].dt.year

# Calculate yearly counts and sums
yearly_admissions = df.groupby('Admission Year').size()
yearly_discharges = df.groupby('Discharge Year').size()
yearly_billing = df.groupby('Admission Year')['Billing Amount'].sum()
yearly_conditions = df.groupby(['Admission Year', 'Medical Condition']).size().unstack(fill_value=0)

# For better visualization, let's sum conditions for each year
yearly_conditions_sum = yearly_conditions.sum(axis=1)


# In[49]:


plt.figure(figsize=(12, 6))
plt.plot(yearly_admissions.index, yearly_admissions.values, marker='o', linestyle='-', label='Admissions', color='blue')
plt.plot(yearly_discharges.index, yearly_discharges.values, marker='o', linestyle='-', label='Discharges', color='red')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title('Yearly Admissions and Discharges')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.show();


# In[52]:


# Plotting total count of medical conditions per year using a stacked bar chart
yearly_conditions.plot(kind='bar', stacked=True, figsize=(14, 8))
plt.title('Yearly Medical Conditions Breakdown')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Medical Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show();


# In[54]:


# Calculate yearly counts
yearly_admissions = df.groupby('Admission Year').size()
yearly_discharges = df.groupby('Discharge Year').size()
yearly_billing = df.groupby('Admission Year')['Billing Amount'].sum()

# Plot yearly admissions and discharges
plt.figure(figsize=(12, 6))
plt.plot(yearly_admissions.index, yearly_admissions.values, marker='o', linestyle='-', label='Admissions', color='blue')
plt.plot(yearly_discharges.index, yearly_discharges.values, marker='o', linestyle='-', label='Discharges', color='red')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title('Yearly Admissions and Discharges')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(yearly_admissions.index)  # Ensure all years are labeled
plt.legend()
plt.show();





# In[55]:


# Plot yearly billing amount
plt.figure(figsize=(12, 6))
plt.plot(yearly_billing.index, yearly_billing.values, marker='o', linestyle='-', label='Billing Amount', color='green')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title('Yearly Billing Amount')
plt.xlabel('Year')
plt.ylabel('Total Billing Amount')
plt.xticks(yearly_billing.index)  # Ensure all years are labeled
plt.legend()
plt.show();


# LINEAR REGRESSION MODEL

# In[59]:


# Extract Length of Stay
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

#  Create Age Group Column
def categorize_age(Age):
    if Age < 18:
        return 'Child'
    elif 18 <= Age < 30:
        return 'Young Adult'
    elif 30 <= Age < 50:
        return 'Adult'
    elif 50 <= Age < 65:
        return 'Middle Aged'
    else:
        return 'Senior'

df['Age Group'] = df['Age'].apply(categorize_age);
print(df['Age Group'] );


# In[60]:


# TRANSFORMER: Encode Categorical Variables using pandas (pd.dummies)
# Use one-hot encoding for 'Medical Condition', 'Age Group', 'Gender', and other categorical columns
categorical_columns = ['Medical Condition', 'Age Group', 'Gender', "Admission Type"]
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Display the first few rows to check the new columns
print(df_encoded.head())


# In[64]:


# Use OneHotEncoder to one-hot encode categorical columns
from sklearn.preprocessing import OneHotEncoder
categorical_columns = ['Medical Condition', 'Age Group', 'Gender']
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# TRANSFORMER: Encode Categorical Variables using sklearn
#Fit and transform the encoder on the data
encoded_data = one_hot_encoder.fit_transform(df[categorical_columns])

# Convert the resulting array back to a DataFrame with proper column names
encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(categorical_columns))

# Concatenate with the original DataFrame (excluding the original categorical columns)
df_encoded = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

# Display the first few rows to check the new columns
print(df_encoded.head());


# In[71]:


# Features and target for predicting Billing Amount
target_billing = "Billing Amount"
features_billing = ['Age Group', 'Medical Condition']

# Features and target for predicting Length of Stay
target_los = "Length of Stay"
features_los = ['Age', 'Medical Condition', 'Admission Type']


# In[72]:


# Split the data for Billing Amount prediction
X_billing = df[features_billing]
y_billing = df[target_billing]
X_train_billing, X_test_billing, y_train_billing, y_test_billing = train_test_split(X_billing, y_billing, test_size=0.2, random_state=42)

# Split the data for Length of Stay prediction
X_los = df[features_los]
y_los = df[target_los]
X_train_los, X_test_los, y_train_los, y_test_los = train_test_split(X_los, y_los, test_size=0.2, random_state=42)


# In[75]:


# Split the data for Billing Amount prediction
X_billing = df[features_billing]
y_billing = df[target_billing]
X_train_billing, X_test_billing, y_train_billing, y_test_billing = train_test_split(X_billing, y_billing, test_size=0.2, random_state=42)

# Split the data for Length of Stay prediction
X_los = df[features_los]
y_los = df[target_los]
X_train_los, X_test_los, y_train_los, y_test_los = train_test_split(X_los, y_los, test_size=0.2, random_state=42)


# In[78]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer

# Load your dataset
df = pd.read_csv(r'C:\Users\ghode\OneDrive - NHS\Desktop\healthcare_dataset.csv')

# Ensure the 'Date of Admission' and 'Discharge Date' columns are in datetime format
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# Feature Engineering: Extract Length of Stay
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# Feature Engineering: Create Age Group Column
def categorize_age(age):
    if age < 18:
        return 'Child'
    elif 18 <= age < 30:
        return 'Young Adult'
    elif 30 <= age < 50:
        return 'Adult'
    elif 50 <= age < 65:
        return 'Middle Aged'
    else:
        return 'Senior'

df['Age Group'] = df['Age'].apply(categorize_age)

# One-Hot Encoding for categorical variables
categorical_columns = ['Medical Condition', 'Age Group', 'Gender', 'Admission Type']
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Features and target for predicting Billing Amount
target_billing = "Billing Amount"
features_billing = ['Age Group', 'Medical Condition']

# Features and target for predicting Length of Stay
target_los = "Length of Stay"
features_los = ['Age', 'Medical Condition', 'Admission Type']

# Split the data for Billing Amount prediction
X_billing = df[features_billing]
y_billing = df[target_billing]
X_train_billing, X_test_billing, y_train_billing, y_test_billing = train_test_split(X_billing, y_billing, test_size=0.2, random_state=42)

# Split the data for Length of Stay prediction
X_los = df[features_los]
y_los = df[target_los]
X_train_los, X_test_los, y_train_los, y_test_los = train_test_split(X_los, y_los, test_size=0.2, random_state=42)

# Pipeline for predicting Billing Amount
pipeline_billing = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_billing)
    ])),
    ('model', LinearRegression())
])

# Pipeline for predicting Length of Stay
pipeline_los = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_los)
    ])),
    ('model', LinearRegression())
])

# Train and evaluate the Billing Amount model
pipeline_billing.fit(X_train_billing, y_train_billing)
y_pred_billing = pipeline_billing.predict(X_test_billing)
mse_billing = mean_absolute_error(y_test_billing, y_pred_billing)
print(f'Mean Absolute Error for Billing Amount: {mse_billing:.2f}')

# Train and evaluate the Length of Stay model
pipeline_los.fit(X_train_los, y_train_los)
y_pred_los = pipeline_los.predict(X_test_los)
mse_los = mean_absolute_error(y_test_los, y_pred_los)
print(f'Mean Absolute Error for Length of Stay: {mse_los:.2f}')


# In[85]:


# Extract coefficients and intercept for Billing Amount model
intercept_billing = pipeline_billing.named_steps['model'].intercept_
coefficients_billing = pipeline_billing.named_steps['model'].coef_
feature_names_billing = pipeline_billing.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(features_billing)


# In[86]:


# Extract coefficients and intercept for Length of Stay model
intercept_los = pipeline_los.named_steps['model'].intercept_
coefficients_los = pipeline_los.named_steps['model'].coef_
feature_names_los = pipeline_los.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(features_los)


# In[87]:


# Billing Amount Regression Equation
print("Billing Amount Regression Equation:")
print(f"y = {intercept_billing:.2f} + ", end="")
for coef, name in zip(coefficients_billing, feature_names_billing):
    print(f"({coef:.2f} * {name}) + ", end="")
print("\b\b ")


# In[88]:


# Length of Stay Regression Equation
print("\nLength of Stay Regression Equation:")
print(f"y = {intercept_los:.2f} + ", end="")
for coef, name in zip(coefficients_los, feature_names_los):
    print(f"({coef:.2f} * {name}) + ", end="")
print("\b\b ")


# In[89]:


def predict_length_of_stay(age, medical_condition, admission_type, model, preprocessor):
    # Create a DataFrame for the new data point
    new_data = pd.DataFrame({
        'Age': [age],
        'Medical Condition': [medical_condition],
        'Admission Type': [admission_type]
    })

    # Preprocess the new data point
    new_data_encoded = preprocessor.transform(new_data)

    # Predict Length of Stay
    predicted_length_of_stay = model.predict(new_data_encoded)

    return predicted_length_of_stay[0]

# Example usage
predicted_los = predict_length_of_stay(50, 'Cancer', 'Elective', pipeline_los.named_steps['model'], pipeline_los.named_steps['preprocessor'])
print(f"Predicted Length of Stay: {predicted_los:.2f} days")


# You can interpret the coefficients to understand the impact of each feature:
# 
# Positive Coefficients: Indicate an increase in Length of Stay when the feature value increases.
# Negative Coefficients: Indicate a decrease in Length of Stay when the feature value increases.
# Detailed Example
# Let's put everything together in a comprehensive manner:
# 
# Communicating Results
# Summary of Findings:
# 
# Age: Patients aged 13 have the highest positive impact on length of stay with a coefficient of 1.99, whereas patients aged 88 and 89 have significant negative impacts with coefficients of -2.67 and -2.60 respectively.
# Medical Conditions: Conditions like asthma have a positive impact (0.26), while conditions like arthritis have a slight negative impact (-0.06).
# Admission Type: Elective admissions slightly increase length of stay (0.06), while urgent admissions decrease it (-0.10).

# In[82]:


# Visualize Feature Importances for Billing Amount
plt.figure(figsize=(10, 6))
plt.barh(feature_names_billing, coefficients_billing)
plt.title('Feature Importances for Billing Amount')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()



# In[84]:


# Visualize Feature Importances for Length of Stay
plt.figure(figsize=(14, 10))  # Increase figure size for better readability
plt.barh(feature_names_los, coefficients_los)
plt.title('Feature Importances for Length of Stay')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')

# Adjust the font size and layout
plt.yticks(fontsize=8)  # Adjust font size as needed
plt.xticks(fontsize=10)
plt.tight_layout()  # Adjust layout to make room for labels

plt.show()


# In[92]:


#function to predict length of stay
def predict_length_of_stay(age, medical_condition, admission_type):
    # Create a DataFrame for the new data point
    new_data = pd.DataFrame({
        'Age': [age],
        'Medical Condition': [medical_condition],
        'Admission Type': [admission_type]
    })

    # Preprocess the new data point
    new_data_encoded = pipeline_los.named_steps['preprocessor'].transform(new_data)

    # Predict Length of Stay
    predicted_length_of_stay = pipeline_los.named_steps['model'].predict(new_data_encoded)

    return predicted_length_of_stay[0]

# Predict Length of Stay for all combinations of age, medical conditions, and admission types
ages = df['Age'].unique()
medical_conditions = df['Medical Condition'].unique()
admission_types = df['Admission Type'].unique()

predictions_los = []

for age in ages:
    for medical_condition in medical_conditions:
        for admission_type in admission_types:
            pred = predict_length_of_stay(age, medical_condition, admission_type)
            predictions_los.append({
                'Age': age,
                'Medical Condition': medical_condition,
                'Admission Type': admission_type,
                'Predicted Length of Stay': pred
            })

# Convert predictions to DataFrame
predictions_los_df = pd.DataFrame(predictions_los)
print(predictions_los_df)



# In[93]:


#function to predict billing amount
def predict_billing_amount(age_group, medical_condition):
    # Create a DataFrame for the new data point
    new_data = pd.DataFrame({
        'Age Group': [age_group],
        'Medical Condition': [medical_condition]
    })

    # Preprocess the new data point
    new_data_encoded = pipeline_billing.named_steps['preprocessor'].transform(new_data)

    # Predict Billing Amount
    predicted_billing_amount = pipeline_billing.named_steps['model'].predict(new_data_encoded)

    return predicted_billing_amount[0]

# Predict Billing Amount for all combinations of age groups and medical conditions
age_groups = df['Age Group'].unique()
medical_conditions = df['Medical Condition'].unique()

predictions_billing = []

for age_group in age_groups:
    for medical_condition in medical_conditions:
        pred = predict_billing_amount(age_group, medical_condition)
        predictions_billing.append({
            'Age Group': age_group,
            'Medical Condition': medical_condition,
            'Predicted Billing Amount': pred
        })

# Convert predictions to DataFrame
predictions_billing_df = pd.DataFrame(predictions_billing)
print(predictions_billing_df)


# In[94]:


# Billing Amount Regression Equation
print("Billing Amount Regression Equation:")
print(f"y = {intercept_billing:.2f}")
for coef, name in zip(coefficients_billing, feature_names_billing):
    print(f" + ({coef:.2f} * {name})")

# Length of Stay Regression Equation
print("\nLength of Stay Regression Equation:")
print(f"y = {intercept_los:.2f}")
for coef, name in zip(coefficients_los, feature_names_los):
    print(f" + ({coef:.2f} * {name})")


# In[96]:


predictions_los_df = pd.DataFrame(predictions_los)
print("Predictions for Length of Stay:")
print(predictions_los_df)


# In[97]:


predictions_billing_df = pd.DataFrame(predictions_billing)
print("Predictions for Billing Amount:")
print(predictions_billing_df)


# In[102]:


get_ipython().system('pip install dash-bootstrap-components')


# In[103]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import dash_bootstrap_components as dbc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

# Load your dataset
df = pd.read_csv(r'C:\Users\ghode\OneDrive - NHS\Desktop\healthcare_dataset.csv')

# Ensure the 'Date of Admission' and 'Discharge Date' columns are in datetime format
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# Feature Engineering: Extract Length of Stay
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# Feature Engineering: Create Age Group Column
def categorize_age(age):
    if age < 18:
        return 'Child'
    elif 18 <= age < 30:
        return 'Young Adult'
    elif 30 <= age < 50:
        return 'Adult'
    elif 50 <= age < 65:
        return 'Middle Aged'
    else:
        return 'Senior'

df['Age Group'] = df['Age'].apply(categorize_age)

# One-Hot Encoding for categorical variables
categorical_columns_billing = ['Medical Condition', 'Age Group']
categorical_columns_los = ['Age', 'Medical Condition', 'Admission Type']
df_encoded_billing = pd.get_dummies(df, columns=categorical_columns_billing)
df_encoded_los = pd.get_dummies(df, columns=categorical_columns_los)

# Train the Billing Amount Model
features_billing = ['Age Group', 'Medical Condition']
target_billing = "Billing Amount"
X_billing = df[features_billing]
y_billing = df[target_billing]

pipeline_billing = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_billing)
    ])),
    ('model', LinearRegression())
])

pipeline_billing.fit(X_billing, y_billing)

# Train the Length of Stay Model
features_los = ['Age', 'Medical Condition', 'Admission Type']
target_los = "Length of Stay"
X_los = df[features_los]
y_los = df[target_los]

pipeline_los = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_los)
    ])),
    ('model', LinearRegression())
])

pipeline_los.fit(X_los, y_los)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Healthcare Predictions Dashboard"),
            html.Label("Age"),
            dcc.Input(id='input-age', type='number', value=30),
            html.Br(),
            html.Label("Medical Condition"),
            dcc.Dropdown(
                id='dropdown-medical-condition',
                options=[{'label': cond, 'value': cond} for cond in df['Medical Condition'].unique()],
                value='Cancer'
            ),
            html.Br(),
            html.Label("Admission Type"),
            dcc.Dropdown(
                id='dropdown-admission-type',
                options=[{'label': adm, 'value': adm} for adm in df['Admission Type'].unique()],
                value='Elective'
            ),
            html.Br(),
            html.Label("Age Group"),
            dcc.Dropdown(
                id='dropdown-age-group',
                options=[{'label': group, 'value': group} for group in df['Age Group'].unique()],
                value='Adult'
            ),
            html.Br(),
            html.Button('Predict', id='button-predict', n_clicks=0)
        ], width=4),
        dbc.Col([
            html.H3("Predicted Length of Stay"),
            html.Div(id='output-len-stay'),
            html.H3("Predicted Billing Amount"),
            html.Div(id='output-bill-amt')
        ], width=8)
    ])
])

@app.callback(
    [Output('output-len-stay', 'children'), Output('output-bill-amt', 'children')],
    [Input('input-age', 'value'), Input('dropdown-medical-condition', 'value'), Input('dropdown-admission-type', 'value'), Input('dropdown-age-group', 'value')]
)
def update_predictions(age, medical_condition, admission_type, age_group):
    # Predict Length of Stay
    new_data_los = pd.DataFrame({
        'Age': [age],
        'Medical Condition': [medical_condition],
        'Admission Type': [admission_type]
    })
    new_data_encoded_los = pipeline_los.named_steps['preprocessor'].transform(new_data_los)
    predicted_length_of_stay = pipeline_los.named_steps['model'].predict(new_data_encoded_los)[0]

    # Predict Billing Amount
    new_data_billing = pd.DataFrame({
        'Age Group': [age_group],
        'Medical Condition': [medical_condition]
    })
    new_data_encoded_billing = pipeline_billing.named_steps['preprocessor'].transform(new_data_billing)
    predicted_billing_amount = pipeline_billing.named_steps['model'].predict(new_data_encoded_billing)[0]

    return f"Predicted Length of Stay: {predicted_length_of_stay:.2f} days", f"Predicted Billing Amount: ${predicted_billing_amount:.2f}"

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




