
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


MODEL_PATH = "new_sevensix.h5"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3"):
    """
    Generate a Grad-CAM heatmap for the predicted class.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Ensure predictions is tensor
        if not isinstance(predictions, tf.Tensor):
            predictions = tf.convert_to_tensor(predictions)

        # Convert predicted class index to Python int
        pred_index = tf.argmax(predictions[0]).numpy()
        class_channel = predictions[:, pred_index]

    # Gradient of predicted class wrt conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize safely
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap.numpy()




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
        heatmap = make_gradcam_heatmap(img_array, model)
        gradcam_img = overlay_heatmap(image, heatmap)
        st.image(gradcam_img, caption="Grad-CAM", use_container_width=True)

       
        st.markdown("""
        **Disclaimer:**
        This tool is for **educational and research purposes only**.
        It is **not a substitute for professional medical advice, diagnosis, or treatment**.
        If you have symptoms or concerns even after a normal X-ray result,
        please consult a qualified healthcare provider.
        """)
