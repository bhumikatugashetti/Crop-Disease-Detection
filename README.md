# 🌿 Crop Disease Detection using Deep Learning

A web-based AI application that detects crop diseases from plant leaf images using a deep learning model built with **TensorFlow/Keras and MobileNetV2**.

The application provides a modern 3D interface where users can upload a plant leaf image and receive a predicted disease along with confidence, disease information, and recommended action.

---

## 🚀 Features

- 🌱 AI-based crop disease detection
- 🧠 MobileNetV2 deep learning architecture
- 📷 Upload plant leaf images for prediction
- 🎯 Confidence score for predictions
- 💡 Disease description and recommended action
- 🖥️ Modern 3D glassmorphism user interface
- 🖱️ Drag-and-drop image upload
- ⚡ Flask-based web application
- 📱 Responsive frontend design
- 🔬 Model trained on 3,317 images

---

## 🧠 Supported Disease Classes

The current model supports the following 5 classes:

| Crop | Disease |
|---|---|
| Apple | Apple Scab |
| Apple | Black Rot |
| Apple | Cedar Apple Rust |
| Apple | Healthy |
| Corn (Maize) | Cercospora Leaf Spot / Gray Leaf Spot |

---

## 🛠️ Technologies Used

### Backend
- Python
- Flask
- TensorFlow
- Keras
- NumPy
- Pillow

### Machine Learning
- MobileNetV2
- Transfer Learning
- Image Classification
- Data Augmentation

### Frontend
- HTML5
- CSS3
- JavaScript
- 3D animations
- Glassmorphism UI

---

## 📂 Project Structure

```text
Crop-Disease-Detection/
│
├── disease_info/
│   └── disease_data.json
│
├── model/
│   ├── class_names.txt
│   └── crop_disease_model.keras
│
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── main.js
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── check_dataset.py
├── convert_model.py
├── model_config.json
├── requirements.txt
├── train_model.py
├── README.md
└── .gitignore
