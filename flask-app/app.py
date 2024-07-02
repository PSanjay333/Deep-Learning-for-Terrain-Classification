from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__, template_folder='templates')

# Load the model
from tensorflow.keras.models import load_model
EfficientNetB0_model = load_model("EfficientnetModel")

# Define class labels
class_labels = ['Airport', 'BareLand', 'Baseball Field', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'Dense Residential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'Medium Residential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port', 'Railway Station', 'Resort', 'River', 'School', 'Sparse Residential', 'Square', 'Stadium', 'Storage Tanks', 'Viaduct']

def preprocess_image(img):
    # Resize image to match model input size
    img = img.resize((224, 224))
    # Convert image to array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

@app.route('/', methods=['GET'])
def upload_form():
    return render_template('upload.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    print(request)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return render_template('error.html',msg='No file selected')

    if file:
        # Open the image file and convert to JPEG format
        img = Image.open(file)
        img = img.convert("RGB")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        # Preprocess the JPEG image
        img = Image.open(img_bytes)
        img_array = preprocess_image(img)
        
        # Make prediction
        prediction = EfficientNetB0_model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]

        # Convert image to base64 string
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        return render_template('result.html', predicted_class=predicted_class_label, img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
