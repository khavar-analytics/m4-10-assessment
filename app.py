from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the true serialized pipeline at server startup
try:
    model_pipeline = joblib.load('penguin_predictor_pipeline.joblib')
    print("SUCCESS: End-to-end pipeline loaded perfectly.")
except Exception as e:
    print(f"ERROR: Could not load the model file. Details: {e}")
    model_pipeline = None

# Matches your original DataFrame format exactly
REQUIRED_FEATURES = [
    'island', 'bill_length_mm', 'bill_depth_mm', 
    'flipper_length_mm', 'body_mass_g', 'sex'
]

@app.route('/health', methods=['GET'])
def health():
    if model_pipeline is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    return jsonify({"status": "unhealthy", "model_loaded": False}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if not model_pipeline:
        return jsonify({"error": "Model pipeline is unavailable."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. Missing JSON body."}), 400

    missing_fields = [field for field in REQUIRED_FEATURES if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

    try:
        # AUTOMATIC STRING NORMALIZATION:
        # Convert strings to standard casing to prevent OneHotEncoder unknown category errors
        # (e.g., "MALE" -> "male" or "Male" depending on your original data configuration)
        island_input = str(data['island']).strip().title() # Transforms "dream" -> "Dream"
        sex_input = str(data['sex']).strip().lower()       # Transforms "MALE" -> "male"
        
        # Structure the input row exactly like your original train data DataFrame
        input_df = pd.DataFrame([{
            'island': island_input,
            'bill_length_mm': float(data['bill_length_mm']),
            'bill_depth_mm': float(data['bill_depth_mm']),
            'flipper_length_mm': float(data['flipper_length_mm']),
            'body_mass_g': float(data['body_mass_g']),
            'sex': sex_input
        }])
        
    except Exception as e:
        return jsonify({"error": f"Failed to parse input data types: {str(e)}"}), 400

    try:
        # Execute prediction using the full structured dataframe pipeline
        prediction = model_pipeline.predict(input_df)[0]
        probabilities = model_pipeline.predict_proba(input_df)[0]
        class_labels = model_pipeline.classes_
        
        prob_mapping = {str(label): round(float(prob), 4) for label, prob in zip(class_labels, probabilities)}

        return jsonify({
            "status": "success",
            "prediction": str(prediction),
            "probabilities": prob_mapping
        }), 200

    except Exception as e:
        # If 'male' fails, let's try 'Male' automatically as a fallback strategy
        try:
            input_df.at[0, 'sex'] = str(data['sex']).strip().title() # Transforms "MALE" -> "Male"
            prediction = model_pipeline.predict(input_df)[0]
            probabilities = model_pipeline.predict_proba(input_df)[0]
            class_labels = model_pipeline.classes_
            
            prob_mapping = {str(label): round(float(prob), 4) for label, prob in zip(class_labels, probabilities)}
            return jsonify({
                "status": "success",
                "prediction": str(prediction),
                "probabilities": prob_mapping
            }), 200
        except:
            return jsonify({"error": f"Inference processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)