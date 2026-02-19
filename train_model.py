import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# =========================
# PATHS (match your structure)
# =========================
DATASET_PATH = "dataset"   # create this folder
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH   = os.path.join(DATASET_PATH, "val")

# =========================
# PARAMETERS
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# =========================
# TRANSFER LEARNING MODEL
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze base model
base_model.trainable = False

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# =========================
# SAVE MODEL
# =========================
model.save("healthy_vs_rotten.h5")

print("\n✅ Model trained and saved as healthy_vs_rotten.h5")
