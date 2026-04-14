
# 🧘‍♀️ Yoga Posture Detection using Deep Learning

## 📌 Project Overview

This project is a **Yoga Posture Detection System** that uses **Deep Learning (ResNet-18)** to identify yoga poses from images. Users can upload an image or capture a live photo using a webcam, and the system predicts the yoga posture with confidence.

---

## 🚀 Features

* 📷 Upload yoga posture images
* 🎥 Capture images using live camera
* 🤖 AI-based posture recognition

* 📊 Displays prediction with confidence
* ⚡ Fast and user-friendly interface
* ❌ Handles unknown/untrained poses

---

## 🛠️ Technologies Used

### 🔹 Backend

* Python
* Flask
* PyTorch
* Torchvision
* timm

### 🔹 Frontend

* HTML
* CSS
* JavaScript

### 🔹 Libraries

* NumPy
* Pillow (PIL)

---

## 📂 Project Structure

```
project/
│── app.py                # Flask backend
│── resNet_18_model.pth  # Trained model
│── labels.json          # Class labels
│── templates/
│   └── index.html       # Frontend UI
│── static/              # (Optional) CSS/JS files
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/yoga-posture-detection.git
cd yoga-posture-detection
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Requirements

```bash
pip install flask torch torchvision timm numpy pillow
```

---

## ▶️ How to Run the Project

```bash
python app.py
```

Then open browser:

```
http://127.0.0.1:5000/
```

---

## 🧠 Model Details

* Model: ResNet-18
* Framework: PyTorch
* Input Size: 224 × 224
* Number of Classes: 23
* Output: Predicted posture with confidence score

---

## 🔄 Working Flow

1. User uploads or captures an image
2. Image is resized to 224×224
3. Image is converted into tensor
4. Model processes the image
5. Prediction is generated
6. Result displayed on UI

---

## ⚠️ Error Handling

* If no image is uploaded → error message
* If confidence < 40% → "Model not trained for this posture"

---

## 📌 Future Improvements

* Add real-time video posture detection
* Improve model accuracy with more data
* Add pose correction suggestions
* Deploy on cloud (AWS/Heroku)

---

## 👩‍💻 Author

* **Bhagyashree Panigrahy**

---

## 📜 License

This project is for educational purposes.

---
