# pneumonia-detection
A ternary classification system using chest X-ray images to detect Pneumonia (classifying as Normal, Bacterial Pneumonia, or Viral Pneumonia) built with a Streamlit web app interface.

## Live Demo
Try it out online: [Live Streamlit App](https://pneumonia-detection-ubmzr64qnwp3x4nthx8d84.streamlit.app/) :contentReference[oaicite:0]{index=0}

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview
This project implements a deep learning model to classify chest X-ray images into one of three classes—Normal, Bacterial Pneumonia, or Viral Pneumonia—and features a Streamlit-based web interface for easy user interaction and visualization.

## Features
- Multi-class image classification (Normal, Bacterial, Viral)
- Interactive web interface built with Streamlit
- Real-time image upload and classification
- Visual feedback, such as probability scores or attention maps (if included)
- Easy-to-use deployment for demonstration or prototyping

## Architecture
- **Model:** _(e.g., CNN base, transfer learning with ResNet, custom model)_  
- **Training:** Implemented via Jupyter or notebook files (e.g., `classification_code.ipynb`)  
- **Interface:** Streamlit application (`streamlit_app.py`) that loads a pre-trained model (`new_sevensix.h5`) to handle user uploads and display results

## Dataset
- Dataset source (e.g., Kaggle, NIH, RSNA)  
- Data organization (e.g., `train/`, `val/`, `test/`)  
- Number of samples per class  
- Preprocessing steps (resizing, normalization, augmentation, etc.)

*(Include specifics or update this section once you have dataset details.)*

## Requirements
Create a `requirements.txt` (already present) containing required packages like:
```bash
streamlit
tensorflow  # or keras
numpy
pillow
...
