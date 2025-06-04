from flask import Flask, request, jsonify, render_template_string
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import os

app = Flask(__name__)

# Model and label encoder setup (same as yours)
class MultiTaskModel(nn.Module):
    def __init__(self, num_flower_classes, num_color_classes):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.flower_head = nn.Linear(in_features, num_flower_classes)
        self.color_head = nn.Linear(in_features, num_color_classes)
        self.oil_head = nn.Linear(in_features, 5)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.flower_head(x), self.color_head(x), self.oil_head(x)

df = pd.read_csv('flowers_labels.csv')
flower_le = LabelEncoder()
color_le = LabelEncoder()
flower_le.fit(df['flower_type'])
color_le.fit(df['dominant_color_label'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskModel(len(flower_le.classes_), len(color_le.classes_))
model_path = 'D:/anki/project/best_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        flower_out, color_out, oil_out = model(x)
        flower_idx = flower_out.argmax(1).item()
        color_idx = color_out.argmax(1).item()
        oil_preds = oil_out.squeeze().cpu().numpy()

    return {
        'predicted_flower_type': flower_le.inverse_transform([flower_idx])[0],
        'predicted_flower_color': color_le.inverse_transform([color_idx])[0],
        'estimated_oil_concentrations': {
            'Linalool': round(float(oil_preds[0]), 4),
            'Geraniol': round(float(oil_preds[1]), 4),
            'Citronellol': round(float(oil_preds[2]), 4)
        }
    }

# Store last prediction here (in-memory)
last_prediction = None

# Main route: show welcome message + upload form + last prediction JSON
@app.route('/', methods=['GET', 'POST'])
def index():
    global last_prediction
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image uploaded", 400
        file = request.files['image']
        save_dir = 'static'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file.filename)
        file.save(save_path)
        try:
            last_prediction = predict_image(save_path)
        except Exception as e:
            last_prediction = {'error': str(e)}
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    # HTML with upload form and JSON display
    return render_template_string('''
        <h1>🌸 Flower Recognition API is running!</h1>
        <h2>Upload a flower image to get prediction:</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" required>
            <input type="submit" value="Predict">
        </form>
        {% if prediction %}
            <h3>Last Prediction Result:</h3>
            <pre>{{ prediction }}</pre>
        {% endif %}
    ''', prediction=last_prediction)

# You can keep your /predict route if you want the pure API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    save_dir = 'static'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    file.save(save_path)
    try:
        result = predict_image(save_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
