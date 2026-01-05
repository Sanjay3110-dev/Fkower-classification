import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

# -------------------- Reproducibility --------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# -------------------- Dataset Paths (GITHUB SAFE) --------------------
train_path = "dataset/train"
val_path = "dataset/val"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 10

print("Checking dataset paths...")
print("Train exists:", os.path.exists(train_path))
print("Val exists:", os.path.exists(val_path))

if not os.path.exists(train_path):
    raise FileNotFoundError("Train folder not found!")
if not os.path.exists(val_path):
    raise FileNotFoundError("Validation folder not found!")

# -------------------- Data Generators --------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nFlower categories:", train_gen.class_indices)

# -------------------- Model --------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------- Callbacks --------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -------------------- Training Phase 1 --------------------
print("\n🔹 Training Phase 1 (Feature Extraction)")
history_1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE_1,
    callbacks=[early_stop]
)

# -------------------- Training Phase 2 (Fine Tuning) --------------------
print("\n🔓 Training Phase 2 (Fine Tuning)")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE_2,
    callbacks=[early_stop]
)

# -------------------- Save Model --------------------
os.makedirs("model", exist_ok=True)
model.save("model/flower_classifier_final.keras")
print("\n✅ MODEL SAVED AS model/flower_classifier_final.keras")

# -------------------- Plot Results --------------------
acc = history_1.history["accuracy"] + history_2.history["accuracy"]
val_acc = history_1.history["val_accuracy"] + history_2.history["val_accuracy"]

loss = history_1.history["loss"] + history_2.history["loss"]
val_loss = history_1.history["val_loss"] + history_2.history["val_loss"]

os.makedirs("outputs", exist_ok=True)

plt.figure()
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig("outputs/accuracy_curve.png")
plt.close()

plt.figure()
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig("outputs/loss_curve.png")
plt.close()

print("📈 Training graphs saved in outputs/")
