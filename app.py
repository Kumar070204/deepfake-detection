from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image
import torch
from torchvision import transforms
from torch import nn
import os
import traceback
from flask_cors import CORS

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepfakeModel(nn.Module):
    def __init__(self):
        super(DeepfakeModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32768, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.flatten(start_dim=1)
        if x.shape[1] != self.fc1.in_features:
            self.fc1 = nn.Linear(x.shape[1], 256).to(x.device)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model = DeepfakeModel()
try:
    model_path = os.path.join(os.path.dirname(__file__), "model_epoch_40.pt")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
except Exception as e:
    print("Failed to load model:", e)
    traceback.print_exc()

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_deepfake(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
    deepfake_prob = probability * 100
    real_prob = (1 - probability) * 100
    return deepfake_prob, real_prob

@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/upload', methods=['GET'])
def serve_upload():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file_path = 'temp_image.jpg'
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        file.save(file_path)
        deepfake_prob, real_prob = predict_deepfake(file_path)
        return jsonify({"deepfake_prob": deepfake_prob, "real_prob": real_prob})
    except Exception as e:
        print("Error during prediction:", e)
        traceback.print_exc()
        return jsonify({"error": "An error occurred during prediction"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8020)
