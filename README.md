# 🧠 Brain Tumor Prediction & Segmentation

This project is a **Computer Vision (CV)** application that combines **Brain Tumor Classification** and **Tumor Segmentation** using deep learning models. It aims to assist medical professionals in the **early detection and localization** of brain tumors through MRI scans.

---

## 📂 Dataset

### 🧩 Classification Dataset
The classification model uses the [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

It contains **Training** and **Testing** images categorized into:

- **Glioma Tumor**  
- **Meningioma Tumor**  
- **Pituitary Tumor**  
- **No Tumor**

### 🎯 Segmentation Dataset
The segmentation model uses the [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).  
It provides MRI images along with their **binary segmentation masks**, highlighting tumor regions.

---

## ⚙️ Project Structure

brain_tumor_prediction/
│── src/
│ ├── data.py # Classification dataset loading & preprocessing
│ ├── model.py # Classification model architecture (ResNet50)
│ ├── train.py # Classification training script
│ │
│ ├── data_segmentation.py # Segmentation dataset loader (images + masks)
│ ├── model_segmentation.py # U-Net-based segmentation model
│ ├── train_segmentation.py # Training script for segmentation model
│ │
│ └── app.py # Streamlit app for classification & segmentation inference
│
│── models/
│ ├── best_resnet50.h5
│ ├── resnet50_brain_tumor.h5
│ ├── best_unet_segmentation.h5
│ ├── training_log.csv
│ └── segmentation_log.csv
│
│── README.md




---

## 🧠 Classification Model

### Architecture
- **Base Model:** ResNet50 (pre-trained on ImageNet)  
- **Fine-tuning:** Last 6 layers unfrozen  
- **Additional Layers:**  
  - Global Average Pooling  
  - Dense layers with dropout  
  - Softmax output (4 classes)

### Training Strategies
- **Data Augmentation** (rotation, flips, zoom, brightness adjustments)  
- **Early Stopping** (patience = 3)  
- **Model Checkpointing** (best validation model)  
- **CSV Logger** for training history  

### 📊 Classification Results
- **Best Validation Accuracy:** 98.4%  
- **Test Accuracy:** 97%  

---

## 🩻 Segmentation Model

### Architecture
- **Model Type:** U-Net (Convolutional Encoder-Decoder)  
- **Encoder:** Based on pre-trained ResNet34 backbone  
- **Decoder:** Up-sampling blocks with skip connections  
- **Loss Function:** Binary Cross-Entropy + Dice Loss  
- **Optimizer:** Adam with learning rate scheduling

### Metrics
| Metric | Score |
|--------|-------|
| **Dice Coefficient** | 0.89 |
| **IoU (Jaccard Index)** | 0.81 |
| **Precision** | 0.92 |
| **Recall** | 0.87 |

### Visualization
The model produces **pixel-level tumor masks** that can be visualized over the original MRI image, highlighting the detected tumor regions.

---

## 🌐 Streamlit Application

An integrated **Streamlit web app** allows you to:
1. Upload an MRI image.
2. View tumor **classification results** (tumor type).
3. Generate and visualize **segmentation masks** for tumor localization.

---

## 🧩 Future Improvements

- Add multi-class segmentation (different tumor types).
- Integrate Grad-CAM for visual explanation of classification.
- Optimize inference pipeline for clinical use.

---

## 🚀 How to Run

```bash
# Train classification model
python src/train.py

# Train segmentation model
python src/train_segmentation.py

# Run the web app
streamlit run src/app.py
