from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Load models for different diseases
MODELS = {
    'heart': joblib.load('models/voting_classifier_Heart.pkl'),
    'liver': joblib.load('models/voting_classifier_Liver.pkl'),
    'kidney': joblib.load('models/voting_classifier_Kidney.pkl'),
    'diabetes':joblib.load('models/voting_classifier_Diabetes.pkl'),
    # 'kidney': joblib.load('models/kidney_disease_model.sav'),
    # 'liver': joblib.load('models/liver_disease_model.sav'),
    # 'stroke': joblib.load('models/stroke_model.sav'),
}

# Feature mappings for each disease
FEATURES = {
    'heart': [
        'age', 'sex', 'chestPainType', 'restingBloodPressure', 'cholesterol',
        'fastingBloodSugar', 'restingECG', 'maxHeartRate', 'exerciseAngina',
        'stDepression', 'stSlope'
    ],
    'diabetes': [
        'pregnancies', 'glucose', 'bloodPressure', 'skinThickness', 'insulin',
        'bmi', 'diabetesPedigree', 'age'
    ],
    'kidney': [
        'age', 'bloodPressure', 'specificGravity', 'albumin', 'sugar',
        'redBloodCells', 'pusCell', 'pusCellClumps', 'bacteria',
        'bloodGlucoseRandom', 'bloodUrea', 'serumCreatinine', 'sodium',
        'potassium', 'hemoglobin', 'packedCellVolume', 'whiteBloodCellCount',
        'redBloodCellCount', 'hypertension', 'diabetesMellitus', 'coronaryArteryDisease',
        'appetite', 'pedalEdema', 'anemia'
    ],
    'liver': [
        'age', 'gender', 'totalBilirubin', 'directBilirubin', 'alkalinePhosphotase',
        'alamineAminotransferase', 'aspartateAminotransferase', 'totalProteins',
        'albumin', 'albuminGlobulinRatio'
    ],
    'stroke': [
        'age', 'hypertension', 'heartDisease', 'everMarried', 'workType',
        'residenceType', 'avgGlucoseLevel', 'bmi', 'smokingStatus'
    ],
}

@app.route('/')
def home():
    """Serve the main index page."""
    return render_template('index.html')

@app.route('/<disease>')
def disease_page(disease):
    """Serve the HTML page for the selected disease."""
    if disease not in MODELS:
        return "Disease not supported.", 404
    return render_template(f'{disease}-disease.html')

@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):
    """Handle predictions for the selected disease."""
    if disease not in MODELS:
        return jsonify({"error": "Disease not supported."}), 404

    # Get the model and features for the disease
    model = MODELS[disease]
    required_features = FEATURES[disease]

    # Parse input data
    data = request.json
    cleaned_data = {}
    for feature in required_features:
        value = data.get(feature, None)
        if value is None or value == "":
            logging.warning(f"Missing value for feature: {feature}")
            return jsonify(
                {"error": f"Missing or invalid value for feature: {feature}"}
            ), 400
        try:
            cleaned_data[feature] = float(value)
        except ValueError:
            logging.error(f"Invalid value for {feature}: {value}")
            return jsonify({"error": f"Invalid value for {feature}: {value}"}), 400

    # Convert to numpy array
    try:
        input_data = np.array([list(cleaned_data.values())]).reshape(1, -1)
        logging.debug(f"Input data for {disease}: {input_data}")
    except Exception as e:
        logging.error(f"Error processing input data for {disease}: {str(e)}")
        return jsonify({"error": f"Error processing input data: {str(e)}"}), 500

    # Apply scaler for diabetes only
    if disease == "diabetes":
        try:
            with open("models/scaler.pkl", "rb") as scaler_file:
                scaler = joblib.load(scaler_file)
            input_data = scaler.transform(input_data)
            logging.debug(f"Scaled input data for liver disease: {input_data}")
        except Exception as e:
            logging.error(f"Error loading scaler for liver disease: {str(e)}")
            return jsonify({"error": f"Error loading scaler: {str(e)}"}), 500

    # Make prediction
    try:
        prediction = model.predict(input_data)
        prediction = float(prediction)  # Convert to python float
        # probability = (
        #     model.predict_proba(input_data)[0][1]
        #     if hasattr(model, "predict_proba")
        #     else None
        # )
        # if probability is not None:
        #     probability = float(probability)  # Convert to python float
        logging.debug(f"Prediction: {prediction}")
    except Exception as e:
        logging.error(f"Error during prediction for {disease}: {str(e)}")
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    # Prepare response
    if disease == 'kidney':
        if prediction == 0:
            response = "Prediction result : POSITIVE, The model predicts that you have Chronic "+disease+" disease" 
        else:
            response = "Prediction result : NEGATIVE, The model predicts that you DO NOT have Chronic"+disease+" disease"
    elif disease == 'diabetes':
        if prediction == 1:
            response = "Prediction result : POSITIVE, The model predicts that you have "+disease
        else:
            response = "Prediction result : NEGATIVE, The model predicts that you DO NOT have "+disease
    else:
        if prediction == 1:
            response = "Prediction result : POSITIVE, The model predicts that you have "+disease+" disease" 
        else:
            response = "Prediction result : NEGATIVE, The model predicts that you DO NOT have "+disease+" disease"
            # "probability": probability,
        
    logging.debug(f"Response for {disease}: {response}")
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)