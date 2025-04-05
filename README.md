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
``bash
conda create -n chestxray-vit python=3.8
conda activate chestxray-vit
pip install -r requirements.txt

## 📥 Download Dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/

##🏋️ Training
bash
Copy
Edit
python train.py \
    --image_size 224 \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4
    
##🔎 Inference
python
Copy
Edit
from inference import predict

prob = predict("patient_123.png")
print(f"Pneumonia Probability: {prob:.2%}")    

##🏗️ Project Structure
csharp
Copy
Edit
├── configs/               # Hyperparameter configurations
│   ├── base.yaml          
│   └── large_model.yaml
├── data/                  # Dataset processing
│   ├── preprocessing.py
│   └── augmentation.py
├── models/                # Model architectures
│   ├── vit.py             # Custom ViT implementation
│   └── losses.py          # Hybrid loss functions
├── notebooks/             # Jupyter notebooks
│   ├── EDA.ipynb          # Exploratory data analysis
│   └── Attention_Vis.ipynb
├── train.py               # Training script
├── inference.py           # Deployment-ready prediction
└── evaluate.py            # Performance metrics
    
