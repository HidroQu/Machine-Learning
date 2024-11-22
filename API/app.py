from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


app = Flask(__name__)

nutrient_model = load_model('model/nutrient.h5')
CLASS_NAMES_NUTRIENT = ['Healthy', 'Kalium_Deficiency',
                        'Nitrogen_Deficiency', 'Phosphorus_Deficiency']
nutrient_model.make_predict_function()

plant_model = load_model('model/plants_mobileNetV2.h5')
CLASS_NAMES_PLANT = ['Bayam', 'Kangkung',
                     'Selada', 'Timun', 'Tomat', 'bokChoy']
plant_model.make_predict_function()


# Preprocesses the input image for prediction.
def preprocess_image(image):
    i = image.load_img(image, target_size=(224, 224), color_mode='rgb')
    i = np.array(i)
    i /= 255.0
    i = np.expand_dims(i, axis=0)

    return i


# Predicts the nutrient deficiency class of the input image.
def predict_image_nutrient(image):
    p = nutrient_model.predict(image)
    p = np.argmax(p, axis=1)
    confidence = float(np.max(predictions[0]))

    return CLASS_NAMES_NUTRIENT[i[0]], confidence


# Predicts the plant type class of the input image.
def predict_image_plant(image):
    p = plant_model.predict(image)
    p = np.argmax(p, axis=1)
    confidence = float(np.max(predictions[0]))

    return CLASS_NAMES_PLANT[i[0]], confidence


# Endpoint to predict nutrient deficiency from an uploaded image.
@app.route('/predictNutrient', method=['POST'])
def predict_nutrient():
    try:
        file = request.files['file']

        if not file:
            return jsonify({"No File Uploaded"})

        image = Image.open(file)
        processed_image = preprocess_image(image)
        predictions, confidence = predict_image_nutrient(processed_image)

        return jsonify({
            'predicted_label': predictions,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'Error': str(e)}), 500


# Endpoint to predict plant type from an uploaded image.
@app.route('/predictPlants', method=['POST'])
def predict_plants():
    try:
        file = request.files['file']

        if not file:
            return jsonify({"No File Uploaded"})

        image = Image.open(file)
        processed_image = preprocess_image(image)
        predictions, confidence = predict_image_plant(processed_image)

        return jsonify({
            'predicted_label': predictions,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'Error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
