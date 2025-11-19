# non-contact-vital-sign-sys

A contactless RGB-D based system for identity authentication, age/gender estimation, height/weight measurement, and pain-site marking designed for outpatient workflow applications.

This repository accompanies the thesis:

**“A Non-contact Multimodal Sensing and Vital Sign Estimation System for Outpatient Workflow Applications”**

---

## 🌟 Overview

This project integrates Microsoft Kinect RGB-D sensing with deep learning-based estimation modules to provide a fully contactless outpatient check-in and physiological measurement system. The system performs four main functions:

### ✔ Identity Authentication
- Face recognition using ArcFace with a ResNet-50 backbone  
- Depth-based liveness detection to prevent spoof attacks  

### ✔ Age & Gender Estimation
- Vision Transformer (ViT-Base-Patch16-384)  
- Multi-task learning including gender classification, age regression, and ordinal age estimation (CORAL)  

### ✔ Height & Weight Estimation
- MediaPipe Pose for extracting 3D keypoints  
- Point-cloud reconstruction from RGB-D  
- Volume estimation with voxelization, PCA ellipsoid, and convex hull  
- Adaptive density model for body weight estimation  

### ✔ Pain-site Marking
- MediaPipe Hands and Pose  
- Fingertip pointing vector and region intersection  
- Depth-based front/back discrimination  
- Structured pain-region output  

The entire measurement process is designed to complete within **15–20 seconds**, suitable for clinical outpatient workflows.

---

## 📁 Repository Structure

```txt
non-contact-vital-sign-sys/
│
├── README.md
├── requirements.txt
├── LICENSE
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
