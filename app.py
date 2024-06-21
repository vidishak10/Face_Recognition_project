
import os
import cv2
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for
import face_recognition
import base64
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
TRAIN_FOLDER = 'train_faces/'
ENCODINGS_FILE = 'encodings.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the upload and result directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize known faces
known_face_encodings = []
known_face_names = []


def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)


def save_encodings(encodings, names, filename):
    with open(filename, 'wb') as f:
        pickle.dump((encodings, names), f)


def load_encodings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_known_faces(folder_path, encodings_file):
    global known_face_encodings, known_face_names
    if os.path.exists(encodings_file):
        known_face_encodings, known_face_names = load_encodings(encodings_file)
        print(f"Loaded encodings from {encodings_file}")
    else:
        for person_name in os.listdir(folder_path):
            person_folder = os.path.join(folder_path, person_name)
            for filename in os.listdir(person_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_folder, filename)
                    image = face_recognition.load_image_file(image_path)
                    image = resize_image(image)  # Resize the image
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person_name)
                        print(f"Loaded encoding for {person_name} from {filename}")
                    else:
                        print(f"No faces found in {filename}")
        save_encodings(known_face_encodings, known_face_names, encodings_file)
        print(f"Saved encodings to {encodings_file}")


# Load the known faces when the server starts
load_known_faces(TRAIN_FOLDER, ENCODINGS_FILE)


def recognize_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request contains a file part
    if 'file' in request.files:
        file = request.files['file']

        # Check if no file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Process the uploaded file
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the uploaded image and return the result
            return process_image(file_path)

    # If no file part in the request, check if imageData is present (for real-time upload)
    elif 'imageData' in request.form:
        # Process the image data
        image_data = request.form['imageData']
        # Save the image data to a file
        filename = 'realtime_image.png'  # or use any desired filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(file_path, 'wb') as f:
            f.write(base64.b64decode(image_data.split(',')[1]))

        # Process the uploaded image and return the result
        return process_image(file_path)

    return jsonify({'error': 'Invalid request'}), 400


def process_image(file_path):
    # Process the image and get the result
    face_locations, face_names = recognize_faces(file_path)

    result_image = cv2.imread(file_path)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(result_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    result_filename = "result_" + os.path.basename(file_path)
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, result_image)

    # Redirect to the result page
    return redirect(url_for('display_result', filename=result_filename, names=','.join(face_names)))


@app.route('/results/<filename>')
def display_result(filename):
    names = request.args.get('names', '').split(',')
    return render_template('result.html', filename=filename, names=names)


@app.route('/realtime')
def realtime():
    return render_template('realtime.html')


@app.route('/results/<result_name>')
def show_result(result_name):
    return render_template('result.html', result_name=result_name)


if __name__ == '__main__':
    app.run(debug=True)

import os
import cv2
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for
import face_recognition
import base64
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
TRAIN_FOLDER = 'train_faces/'
ENCODINGS_FILE = 'encodings.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the upload and result directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize known faces
known_face_encodings = []
known_face_names = []


def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)


def save_encodings(encodings, names, filename):
    with open(filename, 'wb') as f:
        pickle.dump((encodings, names), f)


def load_encodings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_known_faces(folder_path, encodings_file):
    global known_face_encodings, known_face_names
    if os.path.exists(encodings_file):
        known_face_encodings, known_face_names = load_encodings(encodings_file)
        print(f"Loaded encodings from {encodings_file}")
    else:
        for person_name in os.listdir(folder_path):
            person_folder = os.path.join(folder_path, person_name)
            for filename in os.listdir(person_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_folder, filename)
                    image = face_recognition.load_image_file(image_path)
                    image = resize_image(image)  # Resize the image
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person_name)
                        print(f"Loaded encoding for {person_name} from {filename}")
                    else:
                        print(f"No faces found in {filename}")
        save_encodings(known_face_encodings, known_face_names, encodings_file)
        print(f"Saved encodings to {encodings_file}")


# Load the known faces when the server starts
load_known_faces(TRAIN_FOLDER, ENCODINGS_FILE)


def recognize_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request contains a file part
    if 'file' in request.files:
        file = request.files['file']

        # Check if no file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Process the uploaded file
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the uploaded image and return the result
            return process_image(file_path)

    # If no file part in the request, check if imageData is present (for real-time upload)
    elif 'imageData' in request.form:
        # Process the image data
        image_data = request.form['imageData']
        # Save the image data to a file
        filename = 'realtime_image.png'  # or use any desired filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(file_path, 'wb') as f:
            f.write(base64.b64decode(image_data.split(',')[1]))

        # Process the uploaded image and return the result
        return process_image(file_path)

    return jsonify({'error': 'Invalid request'}), 400


def process_image(file_path):
    # Process the image and get the result
    face_locations, face_names = recognize_faces(file_path)

    result_image = cv2.imread(file_path)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(result_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    result_filename = "result_" + os.path.basename(file_path)
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, result_image)

    # Redirect to the result page
    return redirect(url_for('display_result', filename=result_filename, names=','.join(face_names)))


@app.route('/results/<filename>')
def display_result(filename):
    names = request.args.get('names', '').split(',')
    return render_template('result.html', filename=filename, names=names)


@app.route('/realtime')
def realtime():
    return render_template('realtime.html')


@app.route('/results/<result_name>')
def show_result(result_name):
    return render_template('result.html', result_name=result_name)


if __name__ == '__main__':
    app.run(debug=True)

