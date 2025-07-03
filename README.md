
# Pneumonia Detection using CNN

This project implements a Convolutional Neural Network (CNN) based approach for detecting pneumonia in chest X-ray images. The model classifies images into **Normal**, **Bacterial Pneumonia**, and **Viral Pneumonia**, leveraging transfer learning, data augmentation, and other training optimization techniques.

## 📂 Dataset

The dataset used is the **Chest X-Ray Images (Pneumonia)** dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). It contains:
- `train/`: Training images organized into subfolders (`NORMAL`, `PNEUMONIA`)
- `val/`: Validation set
- `test/`: Test set

To run this project, download the dataset and place it under the `data/` directory as follows:

```text
data/
└── chest_xray/
    ├── train/
    ├── val/
    └── test/
```

---

## 🚀 Features

- **Transfer Learning**: Uses pre-trained `VGG16` with frozen convolutional base.
- **Custom Head**: A new dense classifier head is trained on top.
- **Data Augmentation**: Improves generalization using horizontal flip, zoom, shift, etc.
- **Class Relabeling**: Differentiates between `bacterial` and `viral` pneumonia using filename metadata.
- **Early Stopping**: Prevents overfitting during training.
- **Model Export**: Best model is saved as `best_model.h5`.

---

## 🧠 Model Architecture

- Base model: `VGG16` (pre-trained on ImageNet)
- Global Average Pooling
- Fully Connected Layer with Dropout
- Final Softmax layer (3 classes)

---

## 📊 Results

Evaluation is performed on the test set using:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Sample Predictions Visualization

---

## 📁 File Structure
```
├── data/
│ └── chest_xray/
├── notebooks/
│ ├── 1_preprocessing.ipynb
│ ├── 2_model_building.ipynb
│ ├── 3_training.ipynb
│ └── 4_evaluation.ipynb
├── best_model.h5
├── utils.py
└── README.md
```


---


