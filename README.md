# 🧠 Smart Sorting: Transfer Learning for Identifying Rotten Fruits & Vegetables

Smart Sorting is an AI-based web application that automatically detects whether fruits and vegetables are fresh or rotten using deep learning and transfer learning techniques.The system uses a pre-trained convolutional neural network (CNN) model fine-tuned on a dataset of fruits and vegetables. Users can upload an image through a web interface, and the system predicts its condition instantly.

This project aims to improve quality control in agriculture, food industries, supermarkets, and supply chains.

---

## 🎯 Project Objectives

* To build an AI-powered classification system for fresh vs rotten produce
* To apply transfer learning using a pretrained CNN model
* To develop an interactive web interface using Flask
* To automate quality inspection in agriculture and food supply chains
* To follow professional GitHub and version control practices

---

## 🚀 Features

* Upload image of fruit or vegetable
* AI-based freshness detection (Fresh / Rotten)
* Real-time prediction results
* User Registration & Login system
* Secure password storage
* Simple and user-friendly interface
* Local database support
* Easy to extend for more categories

---

## 🥕 Supported Categories

* Apple
* Carrot
* Pepper
* (Fresh and Rotten classes)
Note: More fruits and vegetables can be added by retraining the model.

---

## 🛠️ Technologies Used

- Python
- Flask (Web Framework)
- TensorFlow / Keras
- OpenCV
- NumPy
- SQLite Database
- HTML, CSS, Bootstrap
- Transfer Learning (Pretrained CNN)
- Git & GitHub

---

## 🧩 How It Works

1. User registers and logs in  
2. Uploads an image of a fruit or vegetable  
3. Image is preprocessed  
4. AI model analyzes the image  
5. System displays result: Fresh or Rotten  

---

## 📂 Project Structure
Smart-Sorting/

├── app.py
├── train_model.py
├── database.db
├── static/
│   └── uploads/
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── upload.html
│   └── result.html
├── requirements.txt
├── .gitignore
├── README.md

---

## ▶️ How to Run the Project

### 1. Clone the Repository

git clone https://github.com/ramachandramohan25-web/Smart-Sorting-Transfer-Learning-for-Identifying-Rotten-Fruits-and-Vegetables.git
cd Smart-Sorting-Transfer-Learning-for-Identifying-Rotten-Fruits-and-Vegetables

### 2.Create Virtual Environment (Recommended)

python -m venv venv
Activate virtual environment:
venv\Scripts\activate
You will see (venv) in terminal.

### 3.Install Dependencies

pip install flask tensorflow numpy opencv-python

### 4.Train the Model (Optional)

If the trained model file is not included:
python train_model.py

### 5.Run the Flask Application

python app.py

---

## 🎥 Demo Video

Demo Video Link: https://drive.google.com/file/d/1Odc03Lz6Px9Xyfb9mTgoj4rZhy-82oL5/view?usp=sharing

The demo video shows:
- Application execution
- User registration & login
- Image upload process
- Prediction results

---

## 🔒 Security Practices

* User passwords stored securely using hashing
* Local database protection
* No sensitive information uploaded to repository
* Upload validation for images

---

## 🎯 Use Cases

- Agriculture quality inspection
- Supermarket automation
- Food waste reduction
- Supply chain monitoring
- Smart farming systems

---

## 🧩 Conclusion

Smart Sorting demonstrates how artificial intelligence and transfer learning can be used to automate the detection of rotten fruits and vegetables. The system provides a scalable, efficient, and user-friendly solution for real-world agricultural and food industry challenges.

This project reflects practical AI integration and follows best practices suitable for academic and internship evaluation.

