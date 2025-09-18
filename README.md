# 🧠 Brain Tumor Prediction

This project is a **Computer Vision (CV)** application that leverages **Transfer Learning with ResNet50** to classify MRI scans of the brain into four categories of tumors. The goal is to assist in the early detection of brain tumors using deep learning.

---

## 📂 Dataset

The dataset is sourced from [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

It contains **Training** and **Testing** images categorized into:

- **Glioma Tumor**  
- **Meningioma Tumor**  
- **Pituitary Tumor**  
- **No Tumor**

---

## ⚙️ Project Structure

brain_tumor_prediction/
│── src/
│ ├── data.py # Dataset loading & preprocessing
│ ├── model.py # Model architecture (ResNet50 transfer learning)
│ ├── train.py # Training script with callbacks
│ └── app.py # Streamlit app for inference
│
│── models/ # Saved trained models & logs
│ ├── best_resnet50.h5
│ ├── resnet50_brain_tumor.h5
│ └── training_log.csv
│
│── README.md




---

## 🏗️ Model Architecture

- **Base Model:** ResNet50 (pre-trained on ImageNet)  
- **Fine-tuning:** Last 6 layers unfrozen  
- **Additional Layers:**  
  - Global Average Pooling  
  - Dense layers with dropout  
  - Softmax output (4 classes)  

---

## 🚀 Training

Key training strategies:

- **Data Augmentation** (rotation, flips, zoom, brightness adjustments)  
- **Early Stopping** (patience = 3)  
- **Model Checkpointing** (save best model)  
- **CSV Logger** for training history  

### Example training command:

## 📊 Results

Best Validation Accuracy: 98.4%

Test Accuracy (on validation set): 97%

(These may vary depending on training hardware & random seeds.)   

## 🌐 Streamlit App

A simple Streamlit web app is provided to upload MRI scans and predict tumor type.

