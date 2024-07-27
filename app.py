import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Fungsi untuk Ekstraksi Fitur
def extract_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb_means = cv2.mean(image)[:3]
    hsv_means = cv2.mean(hsv_image)[:3]
    features = np.concatenate((rgb_means, hsv_means))
    return features

# Fungsi untuk Menghitung Tekstur
def calculate_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()
    return texture

# Memuat atau Melatih Model SVM untuk Kematangan Pisang
model_path = 'model/svm_model.pkl'

# Pastikan direktori 'model' ada
os.makedirs(os.path.dirname(model_path), exist_ok=True)

if os.path.exists(model_path):
    model, scaler = joblib.load(model_path)
else:
    data, labels = [], []
    for label, folder in enumerate(['tidak_matang', 'matang']):
        folder_path = os.path.join('data_pisang', folder)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            features = extract_features(image)
            data.append(features)
            labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    joblib.dump((model, scaler), model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deteksi', methods=['POST'])
def deteksi():
    file = request.files['image']
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        features = extract_features(img)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        result = 'Matang' if prediction == 1 else 'Tidak Matang'
        accuracy = model.score(features_scaled, [prediction])
        texture = calculate_texture(img)  # Menghitung tekstur gambar
        threshold = 100  # Tentukan threshold untuk tekstur, sesuaikan nilainya
        texture_result = 'Bertekstur' if texture > threshold else 'Tidak Bertekstur'
        return jsonify(result=result, accuracy=accuracy, texture=texture_result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
