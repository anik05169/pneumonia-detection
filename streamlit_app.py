
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tf_explain.core.grad_cam import GradCAM
import matplotlib.pyplot as plt

MODEL_PATH = "new_sevensix.h5"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)



import numpy as np

def make_gradcam_heatmap(image, model, last_conv_layer_name="block5_conv3", class_index=None):
    # Convert PIL -> NumPy
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Ensure batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    # Predict
    pred = model.predict(image)
    if class_index is None:
        class_index = np.argmax(pred[0])

    # Build grad model
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap
def preprocess_image(image, target_size=(224, 224)):

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # If grayscale, convert to RGB
    if len(image.shape) == 2:  # (H, W)
        image = np.stack((image,)*3, axis=-1)
    elif image.shape[-1] == 1:  # (H, W, 1)
        image = np.concatenate([image]*3, axis=-1)

    # Resize to target size
    image = tf.image.resize(image, target_size).numpy()

    # Normalize
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    original_img = original_img.convert("RGB")  
    original_array = np.array(original_img)
    heatmap = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(original_array, 1-alpha, heatmap, alpha, 0)


st.title("🩺 Pneumonia Classifier (Ternary)")

uploaded_file = st.file_uploader("Upload a Chest X-ray", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Classify"):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0]  

        
        classes = ["Bacterial Pneumonia", "Normal", "Viral Pneumonia"]
        predicted_class = classes[np.argmax(prediction)]

        
        st.subheader(f"Prediction: {predicted_class}")
        st.write("Class Probabilities:")
        for cls, prob in zip(classes, prediction):
            st.write(f"- **{cls}**: {prob:.4f}")

       
        st.markdown("""
        **Confidence** here means the probability assigned by the model for each class.
        The highest probability indicates the model's predicted class.
        """)

       


       
        st.markdown("""
        **Disclaimer:**
        This tool is for **educational and research purposes only**.
        It is **not a substitute for professional medical advice, diagnosis, or treatment**.
        If you have symptoms or concerns even after a normal X-ray result,
        please consult a qualified healthcare provider.
        """)
