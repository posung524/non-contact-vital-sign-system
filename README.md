# non-contact-vital-sign-sys

A contactless RGB-D based system for identity authentication, age/gender estimation, height/weight measurement, and pain-site marking designed for outpatient workflow applications.

This repository accompanies the thesis:

**“A Non-contact Multimodal Sensing and Vital Sign Estimation System for Outpatient Workflow Applications”**

---

## 🌟 System Overview

This project integrates Microsoft Kinect RGB-D sensing with deep learning–based estimation modules to build a fully contactless outpatient check-in and physiological measurement system.

### 🔶 Overall System Architecture
![overall architecture](docs/整體架構圖.png)

---

## 🔍 Module Designs (Architectures)

### 🧩 1. Identity Authentication
![face model](docs/人臉架構.png)

### 🧩 2. Age & Gender Estimation (ViT)
![age gender architecture](docs/age_gender＿架構圖.png)

### 🧩 3. Height & Weight Estimation
![height weight architecture](docs/身高體重_map.png)

### 🧩 4. Pain-site Marking
![pain region architecture](docs/痛痛痛_map.png)

---

## 🚶‍♂️ System Workflow (Chapter 3)
![system workflow](docs/無接觸系統使用流程.png)

---

# 📊 Experimental Results

Below are the results from each module.

---

## 🔹 1. Identity Authentication Results (ArcFace + Liveness)

| ROC Curve | Similarity Histogram | t-SNE Embedding |
|----------|----------------------|------------------|
| ![](results/auth/roc.png) | ![](results/auth/sim_hist.png) | ![](results/auth/tsne.png) |

---

## 🔹 2. Age & Gender Estimation Results (ViT)

### Age Regression Scatter Plot
![age regression scatter](results/age_gender/age_scatter_regression.png)

### Gender Confusion Matrix
![gender confusion matrix](results/age_gender/gender_confusion_matrix.png)

### Gender ROC Curve
![gender roc curve](results/age_gender/gender_roc_curve.png)

---

## 🔹 3. Pain-site Detection Results

### Pain Point Example
![pain point](results/pain_marker/pain_point.png)

### Pain Back / Chest Examples
<div style="display: flex; gap: 10px;">
    <img src="results/pain_marker/pain_back.png" width="45%">
    <img src="results/pain_marker/pain_chest.png" width="45%">
</div>

---

# 📁 Repository Structure

```txt
non-contact-vital-sign-sys/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── auth_module.py
│   ├── age_gender_module.py
│   ├── height_weight_module.py
│   ├── pain_marker_module.py
│   └── common_utils.py
│
├── experiments/
│   ├── configs/
│   └── logs/
│
├── results/
│   ├── auth/
│   ├── age_gender/
│   ├── height_weight/
│   └── pain_marker/
│
└── docs/
