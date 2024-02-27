#!/usr/bin/env python
# coding: utf-8

# # Import libraries
import joblib
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime, timedelta


import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import numpy as np
import itertools

import xgboost as xgbm
from imblearn.over_sampling import SMOTE

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# # Pull all data into df

def process_data():
    soda = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"

    limit = 50000
    offset = 0

    all_data = []

    while True:
        parameters = {
            "$limit": limit,
            "$offset": offset,
        }

        response = requests.get(soda, params=parameters)
        
        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            offset += limit
        else:
            print("Error:", response.status_code, response.text)
            break
            
    df = pd.DataFrame(all_data)

    # # Handle empty data

    ### 1. Filling score

    df['score'] = pd.to_numeric(df['score'], errors='coerce')

    median = df['score'].median()

    # grade to score dict
    grade_to_score = {
        np.nan: median, # median of all records
        'N': median, # median of all records
        'Z': median, # median of all records
        'P': median, # median of all records
        'A': df[(df['score'] >= 0) & (df['score'] < 14)]['score'].median(), # median of not-null 'A' records
        'B': df[(df['score'] >= 14) & (df['score'] < 27)]['score'].median(), # median of not-null 'B' records
        'C': df[df['score'] >= 27]['score'].median() # median of not-null 'C' records
    }

    # apply the dict
    df['score'] = df.apply(
        lambda row: grade_to_score.get(row['grade'], row['score']) if pd.isna(row['score']) else row['score'],
        axis=1
    )

    # for dtype
    df['score'] = df['score'].astype(float)



    """### 2. Filling action and cuisine_description"""

    df['action'].fillna(value='Not Yet Inspected', inplace=True)
    df['cuisine_description'].fillna(value='Other', inplace=True)

    """### 3. Filling everything else"""

    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:  # for numeric-type columns
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
        else: # for string-type columns
            df[column] = df[column].fillna(df[column].mode()[0])

    df.fillna('N/A', inplace=True)
    df.replace(['nan', 'NaN', np.nan], 'N/A', inplace=True)

    # Strip spaces to mitigate negative effects of typos in data entry, etc.
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # # sort by earliest to latest

    df = df.sort_values(by=['inspection_date'])


    # # Agg data

    ### 1. break inspection date

    df['inspection_date'] = pd.to_datetime(df['inspection_date'])
    df['inspection_year'] = df['inspection_date'].dt.year
    df['inspection_month'] = df['inspection_date'].dt.month
    df['inspection_day'] = df['inspection_date'].dt.day

    ### 2. Violation_percentage (the percentage of PAST records of this restaurant having a violation, 
        # out of all its PAST records) -> df['violation_percentage']

    # Create a temp column indicating the current record has a violation
    df['has_violation'] = df['action'].str.contains('Violations were cited').fillna(0).astype(int)

    # Calculate cumulative count of number of inspections/violations up to current record, of this camis(restaurant)
    df['num_inspections'] = df.groupby('camis').cumcount() - 1 # take out current row
    df['cumulative_violations'] = df.groupby('camis')['has_violation'].cumsum() - df['has_violation'] # take out current row

    # Calculate violation percentage
    df.loc[df['num_inspections'] == 0, 'violation_percentage'] = 0 # set the 'violation_percentage' to 0, if there weren't any past violations
    df.loc[df['num_inspections'] != 0, 'violation_percentage'] = df['cumulative_violations'] / df['num_inspections'] # basic cal if there were past violations

    ### 3. Percentage_critical, Percentage_not_critical, Critical_to_non_critical_ratio (out of all PAST records of this restaurant)

    ## Convert 'Not Applicable' to 'Not Critical' 
      #(treating them the same, as Not Applicable usually indicates no violation, which can be translated to "no critical violation" for our study)
    df['critical_flag'] = df['critical_flag'].replace({'Not Applicable': 'Not Critical'})

    # Create binary columns is_critical/is_not_critical for easier calculation, indicating whether the current record has a critical violation or not
    df['is_critical'] = (df['critical_flag'] == 'Critical').astype(int)
    df['is_not_critical'] = (df['critical_flag'] == 'Not Critical').astype(int)

    # Calculate cumulative counts for both cases up to this record, of this camis(restaurant)
    df['num_critical'] = df.groupby('camis')['is_critical'].cumsum() - df['is_critical'] # take out current row
    df['num_not_critical'] = df.groupby('camis')['is_not_critical'].cumsum() - df['is_not_critical'] # take out current row

    # Calculate total inspection count up to this record
    df['total_inspections'] = df['num_critical'] + df['num_not_critical']

    # Calculate the percentages of critical/non-critical up to this record
    df['percentage_critical'] = (df['num_critical'] / df['total_inspections']).fillna(0)
    df['percentage_not_critical'] = (df['num_not_critical'] / df['total_inspections']).fillna(0)

    # get the ratio
    df['critical_to_non_critical_ratio'] = df['num_critical'] / (df['num_not_critical'] + 1)  # Adding 1 to avoid division by zero

    # Drop temp columns
    df.drop(['is_critical', 'is_not_critical', 'total_inspections'], axis=1, inplace=True)


    ### 4. Violation_code_freq & Violation_code_freq_by_camis: get past frequency of the violation code this record has, both in all records and in the past records of only this camis
    # get cumulative count of the violation_code of current record, in all past records

    # get cumulative count of the violation_code of current record, in all past records
    df['violation_code_freq'] = df.groupby(['violation_code']).cumcount()
    df['violation_code_freq'].fillna(0, inplace=True)

    # Inverse transformation
    df['violation_code_freq'] = 1 / (df['violation_code_freq'] + 1e-5)


    # Exponential Decay
    df['violation_code_freq'] = df['violation_code_freq'] * np.exp(-df['violation_code_freq'])


    # get cumulative count of the violation_code of current record, in past records of this current camis(restaurant)
    df['violation_code_freq_by_camis'] = df.groupby(['camis','violation_code']).cumcount()
    df['total_by_camis'] = df.groupby('camis').cumcount()
    df['violation_code_freq_by_camis']= df['violation_code_freq_by_camis']/(df['total_by_camis'])
    df['violation_code_freq_by_camis'].fillna(0, inplace=True)


    """5. the critical_flag and violation_code for the previous inspection of this restaurant"""

    df['prev_violation_code'] = df.groupby('camis')['violation_code'].shift(1).fillna('Unknown')
    df['prev_critical'] = df.groupby('camis')['critical_flag'].shift(1).fillna('Unknown')

    # Remove Outliers for: num_inspections, score, critical_to_non_critical_ratio, num_critical
    #drop the lowest 1% and the highest 99%

    # 1. violation_code_freq
    # 2. score
    # 3. cuisine_description
    # 4. critical_to_non_critical_ratio
    # 5. inspection_year
    low_i = df["num_inspections"].quantile(0.01)
    high_i = df["num_inspections"].quantile(0.99)
    print("1% quantitile for num_inspections: ",(low_i, high_i))
    low_s = df["score"].quantile(0.01)
    high_s = df["score"].quantile(0.99)
    print("1% quantitile for score: ",(low_s, high_s))


    low_r = df["critical_to_non_critical_ratio"].quantile(0.01)
    high_r = df["critical_to_non_critical_ratio"].quantile(0.99)
    print("1% quantitile for critical_to_non_critical_ratio: ",(low_r, high_r))

    low_c = df["num_critical"].quantile(0.01)
    high_c = df["num_critical"].quantile(0.99)
    print("1% quantitile for num_critical: ",(low_c, high_c))

    df = df.loc[(df["num_inspections"] >= low_i) & (df["num_inspections"] <= high_i )]
    df = df.loc[(df["score"] >= low_s) & (df["score"] <= high_s)]
    df = df.loc[(df["critical_to_non_critical_ratio"] >= low_r) & (df["critical_to_non_critical_ratio"] <= high_r)]
    df = df.loc[(df["num_critical"] >= low_c) & (df["num_critical"] <= high_c)]



    # # keep only selected features + target

    df = df[['score', 'action', 'num_inspections', 'violation_percentage',
             'inspection_year', 'inspection_month',
             'percentage_critical', 'percentage_not_critical',
             'critical_to_non_critical_ratio',
             'boro', 'cuisine_description',
             'dba',
             'prev_violation_code', 'prev_critical',
             'violation_code_freq',
             'violation_code_freq_by_camis',
             'critical_flag']]

    ## Filling all nan's just in case
    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:  # for numeric-type columns
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
        else: # for string-type columns
            df[column] = df[column].fillna("N/A")

    df.replace(['nan', 'NaN', np.nan], 'N/A', inplace=True)

    # save df
    df.to_csv('processed_data.csv', index=False)


# # Label the categoricals

def train_predict():
    df = pd.read_csv('processed_data.csv')

    le_dict = {}

    for col in df.columns:
        if df[col].dtype == 'object' and col != 'critical_flag':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le

    # Save the LabelEncoder dictionary
    joblib.dump(le_dict, 'label_encoders.pkl')

    """# 4. Data splitting and scaling

    ## Use newest record for each restaurant as the testing data; every older record as training data

    Advantages:
    1. Further prevent data leakage
    2. Simulate real-world situation (predicting off of newest records, but training with older data); prevents using future data to predict past events which is not realistic
    3. Potentially captures trends over time
    """

    # convert 'critical_flag' to binary format
    df['critical_flag_binary'] = df['critical_flag'].apply(lambda x: 1 if x == 'Critical' else 0)

    # separate features and target variable
    X = df.drop(['critical_flag', 'critical_flag_binary'], axis=1)
    y = df['critical_flag_binary']

    # identify the latest record for each 'dba'
    latest_records = df.groupby('dba').tail(1)

    # split the data into training and test sets
    test_indices = latest_records.index
    train_df = df.drop(test_indices)
    test_df = df.loc[test_indices]

    # extract features and labels for both training and test sets
    X_train = train_df[X.columns]
    y_train = train_df['critical_flag_binary']
    X_test = test_df[X.columns]
    y_test = test_df['critical_flag_binary']

    # print the proportion
    train_proportion = len(train_df) / len(df)
    test_proportion = len(test_df) / len(df)
    print(f"Training set proportion: {train_proportion:.2f}")
    print(f"Test set proportion: {test_proportion:.2f}")

    # 5. Handle Class Imbalance using Oversampling

    # Initialize SMOTE with desired sampling strategy and random state
    smote = SMOTE(random_state=512)

    # Fit on training data and return oversampled data
    X_train, y_train = smote.fit_resample(X_train, y_train)


    xgb = xgbm.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=15,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        objective='binary:logistic',
        random_state=42
    )

    # Fit XGBoost on the training data
    xgb.fit(X_train, y_train)

    # Make predictions on the training and test sets
    y_pred_train_xgb = xgb.predict(X_train)
    y_pred_test_xgb = xgb.predict(X_test)

    # Check the Accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train_xgb)
    test_accuracy = accuracy_score(y_test, y_pred_test_xgb)

    # Print and save accuracy
    print('XGBoost Train Accuracy Accuracy: ', train_accuracy, '%')
    print('XGBoost Test Accuracy Accuracy: ', test_accuracy, '%')

    # Print classification report:
    print(classification_report(y_test, y_pred_test_xgb))

    # Save the model to a file
    joblib.dump(xgb, 'model.pkl')


## Dag Implementation

default_args = {
    'owner': 'bigapple',
    'depends_on_past': False,
    'email': ['sh4355@columbia.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
}

with DAG(
    'big_apple_bite_safe',
    default_args=default_args,
    description='A DAG to fetch data and train prediction models',
    schedule_interval=timedelta(days=30),
    start_date=datetime(2023, 11, 30),
    catchup=False,
    tags=['dag'],
) as dag:
    
    task_fetch_data = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
    )
    
    task_train_and_predict = PythonOperator(
        task_id='train_predict',
        python_callable=train_predict,
    )
    
    # Define task dependencies
    task_fetch_data >> task_train_and_predict

