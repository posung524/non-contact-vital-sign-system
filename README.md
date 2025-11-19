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

## 🔍 Module Designs

Below are the architectures of each individual module used in the system.

---

## 🧩 1. Identity Authentication

Uses **ArcFace** with a **ResNet-50 backbone** and depth-based liveness detection.

![face model](docs/人臉架構.png)

---

## 🧩 2. Age & Gender Estimation (ViT)

Uses **Vision Transformer (ViT-Base-Patch16-384)** with age regression, gender classification, and ordinal age prediction.

![age gender architecture](docs/age_gender＿架構圖.png)

---

## 🧩 3. Height & Weight Estimation

Uses MediaPipe Pose, 3D reconstruction, and three volume estimators (voxel, PCA ellipsoid, convex hull) with quality-weighted fusion.

![height weight architecture](docs/身高體重_map.png)

---

## 🧩 4. Pain-site Marking

Uses hand/pose landmarks, fingertip direction vectors, and depth-based front/back discrimination.

![pain region architecture](docs/痛痛痛_map.png)

---

## 🚶‍♂️ System Workflow (Chapter 3)

The full user flow includes identity authentication, age/gender estimation, height/weight measurement, and pain-region annotation.

![system workflow](docs/無接觸系統使用流程.png)

---

## 📁 Repository Structure

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
    ├── 整體架構圖.png
    ├── 人臉架構.png
    ├── age_gender＿架構圖.png
    ├── 身高體重_map.png
    ├── 痛痛痛_map.png
    └── 無接觸系統使用流程.png
