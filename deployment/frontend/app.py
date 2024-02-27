from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# load all necessary files upfront
le_dict = joblib.load('label_encoders.pkl')  # LabelEncoders
df = pd.read_csv('processed_data.csv')  # preprocessed data
model = joblib.load('model.pkl')  #  trained model
# sort all restaurant names and pass to html to display so users can copy paste & search
sorted_dba = sorted(df['dba'].astype(str).unique()) # made global to pass to all routes

def dynamic_weighting_historical_data(df, model_features):
    aggregated_record = {}

    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:  # numeric columns
            recent_value = df[column].iloc[-1]
            mean_value = df[column].mean()
            aggregated_record[column] = 0.6 * recent_value + 0.4 * mean_value
        else:  # other columns
            mode_value = df[column].tail(10).mode()[0] # get the mode of most recent 10 recordds
            aggregated_record[column] = mode_value

    # reorder the aggregated record cols to the same as model features
    aggregated_record_ordered = {feature: aggregated_record.get(feature) for feature in model_features}

    return aggregated_record_ordered

def encode_input(input_df):
    # encode categorical columns w/ same label encoder as in training
    for col, le in le_dict.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
    return input_df

def get_latest_record(restaurant_name, df):
    restaurant_df = df[df['dba'] == restaurant_name]

    # select the latest record
    latest_record = restaurant_df.iloc[-1]

    # keep only columns we want to display
    columns_to_keep = ['score', 'inspection_year', 'inspection_month', 
                       'percentage_critical', 'percentage_not_critical', 
                       'critical_to_non_critical_ratio', 'boro', 
                       'cuisine_description',
                       'prev_critical', 'critical_flag']

    latest_record = latest_record[columns_to_keep]

    return latest_record.to_dict()



# API endpoints
@app.route('/')
def index(): # homepage
    return render_template('index.html', sorted_dba=sorted_dba)

@app.route('/predict', methods=['POST'])
def predict():
    dba_input = request.form['dba']
    dba_input = dba_input.upper() # upper case to match, but allow user to input lower

    # get rows of inspections at input restaurant
    restaurant_df = df[df['dba'] == dba_input]
    num_records = len(restaurant_df)

    # show message if no records found
    if restaurant_df.empty:
        return render_template('index.html', prediction_text='No records found for the given restaurant name.')

    # dynamic weighting of past records
    aggregated_record = dynamic_weighting_historical_data(restaurant_df, restaurant_df.columns.tolist())
    aggregated_df = pd.DataFrame([aggregated_record])

    # encode w/ label encoder
    encoded_aggregated_df = encode_input(aggregated_df.drop(['critical_flag'], axis=1))

    # get latest record/past ratios to show insights
    latest_record = get_latest_record(dba_input, df)

    # make prediction using the model
    prediction = model.predict(encoded_aggregated_df)
    prediction_text = 'Critical Violation' if prediction[0] == 1 else 'Not Critical Violation/ No violation'

    # display the result
    return render_template('index.html', dba_input=dba_input, prediction_text=prediction_text, num_records=num_records, sorted_dba=sorted_dba, latest_record=latest_record)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
