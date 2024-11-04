import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load the ResNet50 model pretrained on ImageNet
def load_pretrained_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze the layers except the final one
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the final layer for binary classification (Pneumonia vs Normal)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the input image (resize, normalize)
def transform_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224 (ResNet input size)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)  # Open the image
    return preprocess(image).unsqueeze(0)  # Add batch dimension for model input

# Get the prediction (Normal or Pneumonia)
def get_prediction(image_path, model):
    image_tensor = transform_image(image_path)  # Preprocess the image
    output = model(image_tensor)  # Make prediction
    _, predicted = torch.max(output, 1)  # Get the class (0 = Normal, 1 = Pneumonia)
    return predicted.item()  # Return the predicted class
