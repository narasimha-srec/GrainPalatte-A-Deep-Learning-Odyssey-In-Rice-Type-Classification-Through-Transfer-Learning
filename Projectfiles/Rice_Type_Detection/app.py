from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Placeholder for loading the pre-trained VGG16 model
# Replace 'model_path' with the actual path to your trained model
model = None  # load_model('model_path/vgg16_rice_classifier.h5')
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])

# Directory to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prevention tips for farmers based on rice type
PREVENTION_TIPS = {
    'Arborio': {
        'description': 'Arborio is a short-grain rice used in risotto, requiring high water and nutrient-rich soil.',
        'preventions': [
            'Ensure proper water management with consistent flooding to prevent drought stress.',
            'Use organic fertilizers to maintain soil fertility and avoid chemical buildup.',
            'Monitor for fungal diseases like blast; apply fungicides early if detected.',
            'Rotate crops to prevent soil depletion and reduce pest buildup.'
        ]
    },
    'Basmati': {
        'description': 'Basmati is a long-grain, aromatic rice, sensitive to water and temperature stress.',
        'preventions': [
            'Plant in well-drained, loamy soil to avoid waterlogging.',
            'Use drip irrigation to maintain consistent moisture without excess.',
            'Protect against sheath blight with timely fungicide application.',
            'Avoid excessive nitrogen fertilizers to prevent lodging.'
        ]
    },
    'Ipsala': {
        'description': 'Ipsala is a medium-grain rice, commonly grown in Turkey, adaptable to various conditions.',
        'preventions': [
            'Ensure adequate spacing between plants to reduce competition and disease spread.',
            'Apply balanced fertilizers to support grain filling and yield.',
            'Monitor for pests like stem borers and use integrated pest management.',
            'Maintain field hygiene by removing crop residues to prevent disease.'
        ]
    },
    'Jasmine': {
        'description': 'Jasmine is a fragrant, long-grain rice, thriving in warm, humid climates.',
        'preventions': [
            'Use high-quality seeds to ensure uniform germination and disease resistance.',
            'Control weeds early to prevent competition for nutrients.',
            'Monitor for bacterial leaf blight and apply copper-based bactericides if needed.',
            'Ensure proper drainage to avoid root rot in heavy rains.'
        ]
    },
    'Karacadag': {
        'description': 'Karacadag is a traditional Turkish rice, resilient but susceptible to certain pests.',
        'preventions': [
            'Use resistant varieties to minimize pest and disease impact.',
            'Implement crop rotation with legumes to enhance soil health.',
            'Monitor for rice weevils and use pheromone traps for control.',
            'Avoid late planting to reduce exposure to seasonal pests.'
        ]
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_image(file_path):
    # Load and preprocess image for VGG16
    img = load_img(file_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        img_array = preprocess_image(file_path)
        
        # Placeholder for model prediction
        # Replace with actual model prediction logic
        classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        # prediction = model.predict(img_array)
        # predicted_class = classes[np.argmax(prediction, axis=1)[0]]
        predicted_class = classes[0]  # Placeholder result
        
        # Get description and prevention tips
        description = PREVENTION_TIPS[predicted_class]['description']
        preventions = PREVENTION_TIPS[predicted_class]['preventions']
        
        return render_template('result.html', prediction=predicted_class, 
                             description=description, preventions=preventions, 
                             image_path=file_path)
    
    return render_template('index.html', error='Invalid file format')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)