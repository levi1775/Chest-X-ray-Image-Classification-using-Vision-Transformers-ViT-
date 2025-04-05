# 🩺 Chest X-ray Image Classification using Vision Transformers (ViT)

![ViT X-ray](https://user-images.githubusercontent.com/your-image.png) <!-- Optional visual -->

## 📌 Overview

This project implements a deep learning model for **automated classification of chest X-ray images** using **Vision Transformers (ViT)**. The aim is to accurately classify chest conditions such as **Pneumonia** and **Normal** by leveraging the power of transformer-based architectures in the vision domain.

> 🔍 Motivation: Chest X-rays are widely used for diagnosing respiratory diseases. Automating this process can help radiologists make quicker and more accurate decisions.

---

## 🧠 Model Architecture

This project uses a **Vision Transformer (ViT)** architecture instead of traditional CNNs:

- Image is divided into fixed-size patches.
- Each patch is embedded into a vector and added with positional encoding.
- The transformer encoder learns to extract meaningful representations.
- A classification head predicts the label (e.g., Pneumonia, Normal).

Framework: **PyTorch**  
Pretrained model: ✅ `ViT-B/16` from HuggingFace or Timm (optional fine-tuning)

---

## 📁 Dataset

We used the popular **Chest X-ray dataset from Kaggle**:

📎 [Kaggle Link](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

- Classes: `Normal`, `Pneumonia`
- Total Images: ~5,000
- Train/Val/Test split maintained as per original dataset

---

## 🚀 How to Run

### ⚙️ 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ViT-ChestXray-Classifier.git
cd ViT-ChestXray-Classifier
