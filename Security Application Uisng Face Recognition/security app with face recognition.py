import cv2
import numpy as np
import os
import base64
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024  # Maximum file size

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

facenet = FaceNet()
detector = MTCNN()

def face_detection(image):
    original_img = image
    out = detector.detect_faces(original_img)
    if out:
        x, y, w, h = out[0]['box']
        adjustment_factor_w = 0.2
        adjustment_factor_h = 0.2
        new_x = max(0, x - int(w * adjustment_factor_w))
        new_y = max(0, y - int(h * adjustment_factor_h))
        new_w = min(original_img.shape[1] - new_x, int(w * (1 + 2 * adjustment_factor_w)))
        new_h = min(original_img.shape[0] - new_y, int(h * (1 + 2 * adjustment_factor_h)))
        cropped_face = original_img[new_y:new_y + new_h, new_x:new_x + new_w]
        return cropped_face
    else:
        return None

def preprocess_image(img):
    img = cv2.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)
    return img

def get_face_embeddings(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_detect = face_detection(img)
    if face_detect is None:
        return None
    img1 = preprocess_image(face_detect)
    embeddings = facenet.embeddings(img1)
    return embeddings

def compare_embeddings(user_embedding, stored_embedding):
    threshold = 0.56
    similarity = cosine_similarity(user_embedding, stored_embedding.reshape(1, -1))
    return similarity > threshold

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the uploaded image (e.g., run face recognition)
        embedding_user = get_face_embeddings(file_path)
        if embedding_user is None:
            return jsonify({'error': 'No face detected in user image'})

        image_dir = r"C:/Users/marya/Downloads/image/New folder" #database
        db_image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                          if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        match_found = False
        for db_image_path in db_image_paths:
            embedding_stored = get_face_embeddings(db_image_path)
            if embedding_stored is None:
                continue
            match = compare_embeddings(embedding_user, embedding_stored)
            if match:
                match_found = True
                break

        caption = "Authorized" if match_found else "Unauthorized"
        # Return JSON response with filename and caption
        return jsonify({'filename': filename, 'caption': caption})

    return jsonify({'error': 'Unknown error'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process_webcam', methods=['POST'])
def process_webcam():
    image_data = request.form['image_data']
    image_data = image_data.replace('data:image/png;base64,', '')
    image_bytes = base64.b64decode(image_data)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face_detect = face_detection(img)
    if face_detect is None:
        return jsonify({'error': 'No face detected in webcam image'})

    embedding_user = get_face_embeddings(face_detect)
    if embedding_user is None:
        return jsonify({'error': 'No face detected in user image'})

    image_dir = r"C:/Users/marya/Downloads/image/New folder"  # Change this to the path where your database images are stored
    db_image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                      if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    match_found = False
    for db_image_path in db_image_paths:
        embedding_stored = get_face_embeddings(db_image_path)
        if embedding_stored is None:
            continue
        match = compare_embeddings(embedding_user, embedding_stored)
        if match:
            match_found = True
            break

    caption = "Authorized" if match_found else "Unauthorized"
    return jsonify({'image': base64.b64encode(image_bytes).decode('utf-8'), 'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)
