# 🏥 Chest X-Ray Classification using Vision Transformers (ViT)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)

A state-of-the-art deep learning solution for detecting pneumonia from chest X-rays using Vision Transformers. Achieves **94.6% accuracy** with interpretable attention maps.

![Attention Heatmap](assets/attention_vis.png)  
*Example of ViT attention focus on infected lung regions*

---

## 📌 Key Features
- **ViT Model**: Customizable Vision Transformer architecture  
- **Explainability**: Attention visualization for clinical interpretability  
- **Lightweight**: Optimized for deployment (<500MB)  
- **Reproducible**: Complete training/evaluation pipeline  

---

## 🚀 Quick Start

### 🔧 Prerequisites
```bash
conda create -n chestxray-vit python=3.8
conda activate chestxray-vit
pip install -r requirements.txt
