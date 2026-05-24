import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow_datasets as tfds

IMG_SIZE = 224
MODEL_PATH = "models/mobilenetv2_best.keras"

# On recupere les noms de classes directement depuis tfds
# pour etre sur que l'ordre colle avec l'entrainement
info = tfds.builder("plant_village").info
CLASS_NAMES = info.features["label"].names


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


st.set_page_config(page_title="Detection de maladies des plantes", layout="centered")
st.title("Detection de maladies des plantes")
st.write("Uploadez une photo de feuille pour identifier la maladie.")

model = load_model()

uploaded = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Image uploadee", width='stretch')

    # Preprocess comme a l'entrainement
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = preds[idx]

    # Formater le nom : "Tomato___Early_blight" -> "Tomato - Early blight"
    raw = CLASS_NAMES[idx]
    parts = raw.split("___")
    label = parts[0].replace("_", " ") + " - " + parts[1].replace("_", " ")

    st.subheader("Resultat")
    st.success(f"**{label}**")
    st.metric("Confiance", f"{conf:.1%}")

    # Top 5
    with st.expander("Top 5"):
        top5 = np.argsort(preds)[::-1][:5]
        for rank, i in enumerate(top5, 1):
            r = CLASS_NAMES[i]
            p = r.split("___")
            n = p[0].replace("_", " ") + " - " + p[1].replace("_", " ")
            st.write(f"{rank}. {n} ({preds[i]:.1%})")
