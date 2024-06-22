from flask import Flask, request, jsonify
import numpy as np
import cv2
from deepface import DeepFace
import os

app = Flask(__name__)

def is_high_quality(face_img):
    # Ensure the image is in uint8 format
    if face_img.dtype != np.uint8:
        face_img = (face_img * 255).astype(np.uint8)
    # Convert the image to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Calculate the variance of the Laplacian (focus measure)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Return True if the variance is above the threshold, indicating high quality
    return laplacian_var >= 100

def augment_face_data(face_data):
    # List to store augmented face images
    augmented_faces = [face_data]
    # Rotate the image by given angles and append to the list
    for angle in [10, -10, 20, -20]:
        M = cv2.getRotationMatrix2D((face_data.shape[1] // 2, face_data.shape[0] // 2), angle, 1)
        rotated = cv2.warpAffine(face_data, M, (face_data.shape[1], face_data.shape[0]))
        augmented_faces.append(rotated)
    return augmented_faces

def extract_faces_data_from_video(video_path, detector_backend="retinaface", frame_rate=6):
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    count = 0
    extracted_faces_data = []

    while success:
        if count % frame_rate == 0:
            detected_faces = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backend, enforce_detection=False, anti_spoofing=True)
            if detected_faces:
                for face_info in detected_faces:
                    face_data = face_info['face']
                    if is_high_quality(face_data):
                        augmented_faces = augment_face_data(face_data)
                        extracted_faces_data.extend(augmented_faces)
        success, frame = vidcap.read()
        count += 1

    vidcap.release()
    return extracted_faces_data

def preprocess_for_deepface(img):
    # Ensure the image is in uint8 format
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img

def extract_embeddings(face_data_list, model_name="ArcFace"):
    embeddings_list = []
    for face_data in face_data_list:
        embedding = DeepFace.represent(face_data, model_name=model_name, detector_backend="skip")[0]["embedding"]
        embeddings_list.append(embedding)
    return embeddings_list

def find_face_embedding(img_path, detector_backend='mtcnn'):
    img = cv2.imread(img_path)
    detected_faces = DeepFace.extract_faces(img_path=img_path, detector_backend=detector_backend)
    max_area = 0
    max_face = None

    for face_info in detected_faces:
        facial_area = face_info['facial_area']
        face_area = facial_area['w'] * facial_area['h']
        if face_area > max_area:
            max_area = face_area
            max_face = face_info

    if max_face is not None and 'facial_area' in max_face:
        x, y, w, h = max_face['facial_area']['x'], max_face['facial_area']['y'], max_face['facial_area']['w'], max_face['facial_area']['h']
        face_img = img[y:y+h, x:x+w]
        embedding = DeepFace.represent(face_img, model_name='ArcFace', detector_backend='mtcnn')
        return embedding[0]["embedding"]
    else:
        return None

def calculate_cosine_similarity(embedding1, embedding2):
    # Normalize the embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    # Calculate cosine similarity
    cosine_similarity = np.dot(embedding1, embedding2)
    return cosine_similarity

@app.route('/extract_embeddings_from_video', methods=['POST'])
def extract_embeddings_from_video():
    # Save the uploaded video file
    video_file = request.files['video']
    video_path = os.path.join('/tmp', video_file.filename)
    video_file.save(video_path)

    # Extract faces and embeddings from the video
    extracted_faces_data = extract_faces_data_from_video(video_path)
    processed_faces = [preprocess_for_deepface(face) for face in extracted_faces_data]
    embeddings = extract_embeddings(processed_faces)

    # Remove the temporary video file
    os.remove(video_path)

    return jsonify({'embeddings': embeddings})

@app.route('/extract_embeddings_from_image', methods=['POST'])
def extract_embeddings_from_image():
    # Save the uploaded image file
    image_file = request.files['image']
    image_path = os.path.join('/tmp', image_file.filename)
    image_file.save(image_path)

    # Extract embedding from the image
    embedding = find_face_embedding(image_path)
    os.remove(image_path)

    # Return the embedding or an error message
    if embedding is not None:
        return jsonify({'embedding': embedding})
    else:
        return jsonify({'error': 'No face detected'}), 400

@app.route('/compare_embeddings', methods=['POST'])
def compare_embeddings():
    # Get the embeddings from the request
    data = request.get_json()
    embedding1 = np.array(data['embedding1'])
    embedding2 = np.array(data['embedding2'])

    # Calculate cosine similarity between the embeddings
    similarity = calculate_cosine_similarity(embedding1, embedding2)

    return jsonify({'similarity': similarity})

@app.route('/', methods=['GET'])
def home():
    # Default route for the API... so that it doesnt crash when i access just the port
    return "Welcome to this mini Flask API!"

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
