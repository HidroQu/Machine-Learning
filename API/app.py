from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from flaskext.mysql import MySQL
import io

app = Flask(__name__)

mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'apiml'
app.config['MYSQL_DATABASE_PASSWORD'] = 'ApiML#123'
app.config['MYSQL_DATABASE_DB'] = 'hidroqu'
app.config['MYSQL_DATABASE_HOST'] = '168.138.164.252'
mysql.init_app(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model and class definitions
nutrient_model = load_model('model/nutrients.keras')
CLASS_NAMES_NUTRIENT = ['Healthy', 'Kalium_Deficiency',
                        'Nitrogen_Deficiency', 'Phosphorus_Deficiency']


plant_model = load_model('model/plants.keras')
CLASS_NAMES_PLANTS = ['Bayam', 'Kangkung', 'Selada', 'Timun', 'Tomat', 'bokChoy']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file):
    # Convert FileStorage to BytesIO
    img_bytes = io.BytesIO(file.read())
    
    # Preprocess the image
    img = load_img(img_bytes, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image_nutrient(image):
    predictions = nutrient_model.predict(image)
    pred_index = np.argmax(predictions, axis=1)
    confidence = float(np.max(predictions))
    return CLASS_NAMES_NUTRIENT[pred_index[0]], confidence

def predict_image_plant(image):
    predictions = plant_model.predict(image)
    pred_index = np.argmax(predictions, axis=1)
    confidence = float(np.max(predictions))
    return CLASS_NAMES_PLANTS[pred_index[0]], confidence

@app.route('/predictNutrient', methods=['POST'])
def predict_nutrient():
    try:
        if 'nutrient_img' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['nutrient_img']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        # Process image
        try:
            processed_image = preprocess_image(file)
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 400

        # prediction
        try:
            prediction, confidence = predict_image_nutrient(processed_image)
        except Exception as e:
            return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

        try:
            conn = mysql.connect()
            cursor = conn.cursor()
            cursor.execute("SELECT * from diagnostics WHERE disease_label = %s", (prediction,))
            data = cursor.fetchone()
        except Exception as e:
            return jsonify({"error": f"Error fetch DB: {str(e)}"}), 500

        return jsonify({
            'predicted_label': prediction,
            'confidence': float(confidence),
            'data': data,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/predictPlant', methods=['POST'])
def predict_plant():
    try:
        if 'plant_img' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['plant_img']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        # Process image
        try:
            processed_image = preprocess_image(file)
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 400

        # prediction
        try:
            prediction, confidence = predict_image_plant(processed_image)
        except Exception as e:
            return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

        try:
            conn = mysql.connect()
            cursor = conn.cursor()
            cursor.execute("SELECT * from plants WHERE name LIKE %s", (f"%{prediction}%",))
            data = cursor.fetchone()
        except Exception as e:
            return jsonify({"error": f"Error fetch DB: {str(e)}"}), 500

        return jsonify({       
            'predicted_label': prediction,
            'confidence': float(confidence),
            'data': data,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
