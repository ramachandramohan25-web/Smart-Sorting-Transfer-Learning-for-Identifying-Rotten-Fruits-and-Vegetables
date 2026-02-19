from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import numpy as np
import cv2
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# =========================================================
# 🚀 APP INIT
# =========================================================
app = Flask(__name__)
app.secret_key = "smart_sorting_secret_key"

# =========================================================
# 📁 CONFIG
# =========================================================
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
DATABASE = "database.db"

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

MODEL_PATH = "healthy_vs_rotten.h5"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================================
# 🗄️ DATABASE INIT
# =========================================================
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =========================================================
# 🧠 MODEL TRAINING
# =========================================================
CLASS_NAMES = []   # global

def train_model():
    global CLASS_NAMES

    print("🚀 Training model from dataset folder...")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True
    )

    # 🔥 Save real class order
    CLASS_NAMES = train_ds.class_names
    print("✅ Class mapping:", CLASS_NAMES)

    # 🔥 Normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # transfer learning

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    model.save(MODEL_PATH)
    print("✅ Model trained and saved as healthy_vs_rotten.h5")

    return model

# =========================================================
# 📦 LOAD OR TRAIN MODEL
# =========================================================
if os.path.exists(MODEL_PATH):
    print("✅ Loading existing model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 🔥 Reload class names from dataset
    temp_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True
    )
    CLASS_NAMES = temp_ds.class_names
    print("✅ Class mapping loaded:", CLASS_NAMES)
else:
    model = train_model()

# =========================================================
# 🧠 AI PREDICTION (FIXED VERSION)
# =========================================================
def model_predict(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return "Invalid Image", 0.0

    # ✅ BGR → RGB (CRITICAL FIX)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img = img.astype("float32") / 255.0

    # Expand dims
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img, verbose=0)[0][0]
    print("Raw prediction:", pred)

    # Confidence
    confidence = max(pred, 1 - pred)
    CONFIDENCE_THRESHOLD = 0.65

    if confidence < CONFIDENCE_THRESHOLD:
        return "Invalid Image", confidence

    # 🔥 Correct dynamic mapping
    class_index = int(pred >= 0.5)
    label = CLASS_NAMES[class_index]

    return label.capitalize(), confidence

# =========================================================
# 🏠 HOME
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")

# =========================================================
# 🔐 LOGIN
# =========================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user"] = user[1]
            return redirect(url_for("upload"))
        else:
            error = "Invalid email or password"

    return render_template("login.html", error=error)

# =========================================================
# 📝 REGISTER
# =========================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    success = None

    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DATABASE)
            c = conn.cursor()
            c.execute(
                "INSERT INTO users(username,email,password) VALUES(?,?,?)",
                (username, email, hashed_password)
            )
            conn.commit()
            conn.close()
            success = "Registration successful! Please login."
        except sqlite3.IntegrityError:
            error = "Username or Email already exists"

    return render_template("register.html", error=error, success=success)

# =========================================================
# 🚪 LOGOUT
# =========================================================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# =========================================================
# 📤 UPLOAD + AI
# =========================================================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("upload.html", error="No file selected")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        label, confidence = model_predict(filepath)

        if label.lower() == "fresh":
            result = f"Fresh ✅ ({confidence*100:.2f}%)"
        elif label.lower() == "rotten":
            result = f"Rotten ❌ ({confidence*100:.2f}%)"
        else:
            result = "Invalid Image ❌"

        return render_template("result.html", img_path=filepath, result=result)

    return render_template("upload.html")

# =========================================================
# ▶ RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
