from flask import Flask, request, render_template
from model import load_pretrained_model, get_prediction

app = Flask(__name__)

# Load the pretrained model once when the app starts
model = load_pretrained_model()

@app.route('/')
def index():
    return render_template('index.html')  # Renders homepage for file upload


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        # Save uploaded image to the uploads folder
        file_path = f"uploads/{file.filename}"
        file.save(file_path)
        
        # Get the prediction from the model
        prediction = get_prediction(file_path, model)
        
        # Map the prediction to a label
        label = "Pneumonia" if prediction == 1 else "Normal"
        
        # Display the result page with the prediction
        return render_template('result.html', label=label)

if __name__ == '__main__':
    app.run(debug=True)
import torch
import torchvision.models as models

# Load the saved model

