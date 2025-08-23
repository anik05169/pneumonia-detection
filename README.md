#  Pneumonia Detection (Normal / Bacterial / Viral)

This project implements a **ternary classifier** to detect **Normal, Bacterial Pneumonia, and Viral Pneumonia** from chest X-ray images using **Transfer Learning (VGG16)**. It also provides **explainability** via **Grad-CAM visualizations**, and an interactive **Streamlit web app** for uploading and classifying X-ray images.

---

## Dataset Preparation

1. Download and unzip your dataset into Google Drive.
2. Mount Google Drive in Colab:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Unzip the dataset:

   ```bash
   !unzip -q "/content/drive/MyDrive/images.zip" -d /content/pneumonia_data
   ```
4. Reorganize dataset into three categories (`NORMAL`, `BACTERIAL`, `VIRAL`) inside `train`, `val`, and `test` directories.

---

## Model Architecture

* **Base Model:** VGG16 (pretrained on ImageNet, frozen)
* **Custom Layers:**

  * Flatten
  * Dense (128, ReLU)
  * Dropout (0.5)
  * Dense (3, Softmax)
* **Optimizer:** Adam (lr=1e-4)
* **Loss:** Categorical Crossentropy
* **Metrics:** Accuracy

---

##  Training

```python
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)
```

* **EarlyStopping** with patience=3
* **ModelCheckpoint** to save best model (`best_model.h5`)

---

##  Evaluation

```python
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

print(classification_report(y_true, y_pred, target_names=class_labels))
```

* Generates **Confusion Matrix** and **Classification Report**.

---

## Explainability with Grad-CAM

We use **Grad-CAM** to highlight regions of the X-ray that contributed most to the model's prediction.

```python
from tf_explain.core.grad_cam import GradCAM
explainer = GradCAM()
explanation = explainer.explain(
    validation_data=(image, None),
    model=model,
    class_index=predicted_class,
    layer_name="block5_conv3"
)
```

* Bright (red/yellow) = Important regions
* Cool (blue/green) = Less important regions

---

##  Streamlit Web App

Run locally:

```bash
streamlit run streamli_app.py
```

### Features

* Upload a chest X-ray (JPG/PNG)
* Classify as:

  * **Normal**
  * **Bacterial Pneumonia**
  * **Viral Pneumonia**
* Show **class probabilities**
* Overlay **Grad-CAM heatmap** for interpretability

---

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies:

* TensorFlow / Keras
* NumPy
* Matplotlib
* scikit-learn
* OpenCV
* tf-explain
* Streamlit

---

##  Disclaimer

This project is for **educational and research purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider for medical concerns.

---

## Repository

GitHub: [anik05169/pneumonia-detection](https://github.com/anik05169/pneumonia-detection)

---

## To-Do

* [ ] Improve accuracy with fine-tuning
* [ ] Add more data augmentation
* [ ] Support Grad-CAM on uploaded images
