import io
import json
import os

import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flaskext.mysql import MySQL
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

load_dotenv()

app = Flask(__name__)

mysql = MySQL()

app.config.update(
    MYSQL_DATABASE_HOST=os.getenv('MYSQL_DATABASE_HOST', 'localhost'),
    MYSQL_DATABASE_USER=os.getenv('MYSQL_DATABASE_USER', 'root'),
    MYSQL_DATABASE_PASSWORD=os.getenv('MYSQL_DATABASE_PASSWORD', 'password'),
    MYSQL_DATABASE_DB=os.getenv('MYSQL_DATABASE_DB', 'hidroqu'),
)

mysql.init_app(app)

ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg' }

MODELS = {
    "nutrient": {
        "model": load_model('model/nutrients_saved'),
        "classes": ['Healthy', 'Kalium_Deficiency', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency']
    },
    "plant": {
        "model": load_model('model/plants_saved'),
        "classes": ['Bayam', 'Kangkung', 'Selada', 'Timun', 'Tomat', 'bokChoy']
    }
}


def is_allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file):
    """Convert and preprocess image for prediction."""
    img_bytes = io.BytesIO(file.read())
    img = load_img(img_bytes, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_image(image, model, class_names):
    """Predict the class of an image."""
    predictions = model.predict(image)
    pred_index = np.argmax(predictions, axis=1)
    confidence = float(np.max(predictions))
    return class_names[pred_index[0]], confidence


def fetch_from_db(query, params):
    """Fetch data from the database."""
    try:
        conn = mysql.connect()
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")


def prepare_response(status, data=None, error=None):
    """Prepare a consistent JSON response."""
    response = { "status": status }
    if data:
        response["data"] = data
    if error:
        response["error"] = error
    return jsonify(response)


def handle_prediction(file_key, model_info, db_query, response_formatter):
    """Generic handler for prediction routes."""
    try:
        if file_key not in request.files:
            return prepare_response("error", error="No file part"), 400

        file = request.files[file_key]
        if not file or not file.filename:
            return prepare_response("error", error="No selected file"), 400

        if not is_allowed_file(file.filename):
            return prepare_response("error", error="File type not allowed"), 400

        try:
            processed_image = preprocess_image(file)
            prediction, confidence = predict_image(processed_image, model_info["model"], model_info["classes"])
        except Exception as e:
            return prepare_response("error", error=f"Prediction error: {e}"), 500

        try:
            data = fetch_from_db(db_query, (prediction,))
        except Exception as e:
            return prepare_response("error", error=f"Database fetch error: {e}"), 500

        if not data:
            return prepare_response("error", error="No data found for prediction"), 404

        return response_formatter(prediction, confidence, data)
    except Exception as e:
        return prepare_response("error", error=str(e)), 500


def format_nutrient_response(prediction, confidence, data):
    """Format response for nutrient predictions."""
    return prepare_response(
        "success", {
            "predicted_label": prediction,
            "confidence": confidence,
            "diagnostic": {
                "id": data[0],
                "name_disease": data[1],
                "disease_label": data[2],
                "disease_image": json.loads(data[3]),
                "indication": data[4],
                "cause": data[5],
                "solution": data[6]
            }
        }
        )


def format_plant_response(prediction, confidence, data):
    """Format response for plant predictions."""
    return prepare_response(
        "success", {
            "predicted_label": prediction,
            "confidence": confidence,
            "plant": {
                "id": data[0],
                "name": data[1],
                "latin_name": data[2],
                "icon_plant": data[3],
                "description": data[4],
                "planting_guide": data[5],
                "fertilizer_type": data[6],
                "fun_fact": data[7],
                "duration_plant": data[8]
            }
        }
        )


@app.route('/predictNutrient', methods=['POST'])
def predict_nutrient():
    """Handle nutrient prediction requests."""
    return handle_prediction(
        file_key='nutrient_img',
        model_info=MODELS['nutrient'],
        db_query="SELECT * FROM diagnostics WHERE disease_label = %s",
        response_formatter=format_nutrient_response
    )


@app.route('/predictPlant', methods=['POST'])
def predict_plant():
    """Handle plant prediction requests."""
    return handle_prediction(
        file_key='plant_img',
        model_info=MODELS['plant'],
        db_query="SELECT * FROM plants WHERE name LIKE %s",
        response_formatter=format_plant_response
    )


if __name__ == '__main__':
    app.run(debug=True)
