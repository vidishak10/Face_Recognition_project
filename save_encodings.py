<<<<<<< HEAD
import os
import pickle

import cv2
import face_recognition

TRAIN_FOLDER = 'train_faces/'
ENCODINGS_FILE = 'encodings.pkl'

known_face_encodings = []
known_face_names = []


def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)


def load_known_faces(folder_path):
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


# Encode faces and save to a file
def save_encodings():
    load_known_faces(TRAIN_FOLDER)
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"Encodings saved to {ENCODINGS_FILE}")


if __name__ == '__main__':
    save_encodings()
=======
import os
import pickle

import cv2
import face_recognition

TRAIN_FOLDER = 'train_faces/'
ENCODINGS_FILE = 'encodings.pkl'

known_face_encodings = []
known_face_names = []


def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)


def load_known_faces(folder_path):
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


# Encode faces and save to a file
def save_encodings():
    load_known_faces(TRAIN_FOLDER)
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"Encodings saved to {ENCODINGS_FILE}")


if __name__ == '__main__':
    save_encodings()
>>>>>>> origin/main
