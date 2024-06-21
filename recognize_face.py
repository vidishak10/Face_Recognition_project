<<<<<<< HEAD
import os
from flask import Flask, request, redirect, url_for, render_template
import cv2
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the upload and result directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load class names
with open('models/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f]

# Load the trained model
model = InceptionResnetV1(classify=True, num_classes=len(class_names)).cuda()
model.load_state_dict(torch.load('models/face_recognition_model.pth'))
model.eval()

# Define transformation
transform = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

# Function to recognize faces
def recognize_face(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).cuda()
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        name = recognize_face(file_path)
        result_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)
        image = cv2.imread(file_path)
        cv2.putText(image, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(result_path, image)
        return redirect(url_for('display_result', filename=file.filename, name=name))
    return redirect(request.url)

@app.route('/results/<filename>')
def display_result(filename):
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    name = request.args.get('name', '')
    html = f'<h1>Recognition Result</h1><img src="{url_for("static", filename="results/" + filename)}"><p>Recognized as: {name}</p>'
    return html

if __name__ == '__main__':
    app.run(debug=True)
=======
import os
from flask import Flask, request, redirect, url_for, render_template
import cv2
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the upload and result directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load class names
with open('models/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f]

# Load the trained model
model = InceptionResnetV1(classify=True, num_classes=len(class_names)).cuda()
model.load_state_dict(torch.load('models/face_recognition_model.pth'))
model.eval()

# Define transformation
transform = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

# Function to recognize faces
def recognize_face(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).cuda()
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        name = recognize_face(file_path)
        result_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)
        image = cv2.imread(file_path)
        cv2.putText(image, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(result_path, image)
        return redirect(url_for('display_result', filename=file.filename, name=name))
    return redirect(request.url)

@app.route('/results/<filename>')
def display_result(filename):
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    name = request.args.get('name', '')
    html = f'<h1>Recognition Result</h1><img src="{url_for("static", filename="results/" + filename)}"><p>Recognized as: {name}</p>'
    return html

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> origin/main
