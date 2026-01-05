# 🌸 Flower Classification using MobileNetV2

This project implements an image classification model to identify different types of flowers using **TensorFlow** and **MobileNetV2** with transfer learning.

---

## 🚀 Features
- Uses **MobileNetV2 (pretrained on ImageNet)**
- Data augmentation for better generalization
- Two-phase training:
  - Feature extraction
  - Fine-tuning
- Early stopping to prevent overfitting
- Saves trained model
- Plots accuracy and loss curves

---

## 🧠 Model Architecture
- MobileNetV2 (base model)
- Global Average Pooling
- Dense + Dropout layers
- Softmax output layer

---

## 📂 Dataset Structure
/dataset
    /train
        /daisy
        /dandelion
        /rose
        /sunflower
        /tulip
    /validation
        /daisy
        /dandelion
        /rose
        /sunflower
        /tulip
