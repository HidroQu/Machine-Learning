from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np



app = Flask(__name__)
model = load_model('model/nutrient.h5')
CLASS_NAMES_NUTRIENT = ['Healthy', 'Kalium_Deficiency', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency'] 


model.make_predict_function()

def preprocess_image(image):
  i = image.load_img(image, target_size=(224,224), color_mode='rgb')
  i = np.array(i)
  i /= 255.0
  i = np.expand_dims(i, axis=0)
  
  return i

def predict_image_nutrient(image):
  p = model.predict(image)
  p = np.argmax(p, axis=1)
  confidence = float(np.max(predictions[0]))
  
  return CLASS_NAMES_NUTRIENT[i[0]], confidence
  
  

@app.route('/predictNutrient', method=['POST'])
def predict_nutrient():
  try:
    file = request.files['file']
    
    if not file:
      return jsonify({"No File Uploaded"})
    
    image = Image.open(file)
    processed_image = preprocess_image(image)
    predictions, confidence = predict_image(processed_image)
    
    return jsonify({
      'predicted_label': predictions,
      'confidence': confidence
    })
    
  except Exception as e:
    return jsonify({'Error': str(e)}), 500
  

    
if __name__ == '__main__':
  app.run(debug=True)
  

