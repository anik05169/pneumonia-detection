
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



def make_gradcam_heatmap(image, model):
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    # Predict
    pred = model.predict(image)
    predicted_class = np.argmax(pred[0])  # use first element of batch

    # Grad-CAM
    explainer = GradCAM()
    explanation = explainer.explain(
        validation_data=(image, None),
        model=model,
        class_index=predicted_class,
        layer_name="block5_conv3"
    )
    return explanation







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
        **Grad-CAM** (Gradient-weighted Class Activation Mapping) shows
        which regions of the X-ray most influenced the model’s decision.
        Warmer colors indicate higher importance.
        """)

       
        st.subheader("Grad-CAM Visualization")
        
        heatmap = make_gradcam_heatmap(image, model)
        gradcam_img = overlay_heatmap(image, heatmap)
        st.image(gradcam_img, caption="Grad-CAM", use_container_width=True)

       
        st.markdown("""
        **Disclaimer:**
        This tool is for **educational and research purposes only**.
        It is **not a substitute for professional medical advice, diagnosis, or treatment**.
        If you have symptoms or concerns even after a normal X-ray result,
        please consult a qualified healthcare provider.
        """)
