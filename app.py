from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import firebase_admin
from firebase_admin import db, credentials
from sklearn.exceptions import NotFittedError

cred = credentials.Certificate("credentials1.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://esp8266-957d4-default-rtdb.firebaseio.com/"})

# Firebase references
ref1 = db.reference("/N")
ref2 = db.reference("/P")
ref3 = db.reference("/K")
ref4 = db.reference("/temp_C")
ref5 = db.reference("/humid")

# Load the model
model = pickle.load(open('final_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    firebase_data = {
        'N': ref1.get(),
        'P': ref2.get(),
        'K': ref3.get(),
        'temp_C': ref4.get(),
        'humid': ref5.get(),
    }
    return render_template('index.html', firebase_data=firebase_data)

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        firebase_data = {
            'N': ref1.get(),
            'P': ref2.get(),
            'K': ref3.get(),
            'temp_C': ref4.get(),
            'humid': ref5.get(),
        }

        # Ensure data exists and convert to appropriate types
        if all(value is not None for value in firebase_data.values()):
            N = float(firebase_data['N'])
            P = float(firebase_data['P'])
            K = float(firebase_data['K'])
            temp = float(firebase_data['temp_C'])
            hum = float(firebase_data['humid'])
        else:
            return jsonify({'error': 'Incomplete data from Firebase'}), 400

        # Prepare input data (remove AH if not used in the model)
        input_data = np.array([[N, P, K, temp, hum]])

        # Predict using the model
        try:
            probabilities = model.predict_proba(input_data)[0]  # Get probabilities for all classes
            class_names = model.classes_

            # Create a dictionary of class names and their corresponding probabilities
            class_probabilities = {class_name: prob for class_name, prob in zip(class_names, probabilities)}

            # Sort class names by probability
            class_names_sorted = sorted(class_probabilities, key=class_probabilities.get, reverse=True)[:5]

            # Create a string of suitable crops
            suitable_crops = ', '.join(class_names_sorted).title()

            return f'Top 5 Suitable Crops are: {suitable_crops}'
        except NotFittedError:
            return jsonify({'error': 'Model is not fitted yet.'}), 500
        except AttributeError:
            return jsonify({'error': 'Model does not support predict_proba method.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

data = pd.read_csv('vegetablecropNPK.csv')

# Preprocess the data
data['N'] = data['N'].apply(lambda x: eval(x.replace('-', '+')) / 2)
data['P'] = data['P'].apply(lambda x: eval(x.replace('-', '+')) / 2)
data['K'] = data['K'].apply(lambda x: eval(x.replace('-', '+')) / 2)

# Define the pipeline
categorical_features = ['Vegetables']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])
model_choice = RandomForestRegressor(n_estimators=100, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model_choice)])

# Fit the pipeline
pipeline.fit(data[['Vegetables']], data[['N', 'P', 'K']])

@app.route('/choice', methods=['POST'])
def predict():
    # Get the target crop from the request data
    target_crop = request.json['crop']

    # Predict fertilizer amount for the target crop
    crop_data = data[data['Vegetables'] == target_crop]
    predictions = pipeline.predict(crop_data[['Vegetables']])
    fertilizer_amount = predictions.mean(axis=0)

    # Return the result
    result = {
        'crop': target_crop,
        'N': fertilizer_amount[0],
        'P': fertilizer_amount[1],
        'K': fertilizer_amount[2]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
