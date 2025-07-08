from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from intervention_engine import suggest_intervention

app = Flask(__name__)
model = joblib.load('models/stress_predictor.pkl')

# Home route: serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route: handle POST requests from the form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        recommendation = suggest_intervention(prediction)
        return jsonify({
            'predicted_stress_level': prediction,
            'recommendation': recommendation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

