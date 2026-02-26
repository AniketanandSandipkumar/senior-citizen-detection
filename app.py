import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Senior Citizen Detection", layout="centered")

st.title("🧓 Senior Citizen Detection System")
st.write("Upload an image to detect age, gender, and senior citizen status.")

# Load Models (cached)
@st.cache_resource
def load_models():
    age_model = load_model("face_age.h5", compile=False)
    gender_model = load_model("face_gender.h5", compile=False)
    return age_model, gender_model

age_model, gender_model = load_models()

AGE_CLASS_MAP = {
    0: "0-20",
    1: "21-30",
    2: "31-60",
    3: "60+"
}

GENDER_MAP = {
    0: "Male",
    1: "Female"
}

def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def predict_age(face_img):
    processed = preprocess_face(face_img)
    preds = age_model.predict(processed, verbose=0)
    age_class = np.argmax(preds)
    return AGE_CLASS_MAP.get(age_class, "Unknown")

def predict_gender(face_img):
    processed = preprocess_face(face_img)
    pred = gender_model.predict(processed, verbose=0)
    gender_class = int(pred[0][0] > 0.5)
    return GENDER_MAP.get(gender_class, "Unknown")

def is_senior(age_range):
    return age_range in ["60+"]

# Image Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]

            age_range = predict_age(face_img)
            gender = predict_gender(face_img)
            senior_flag = "YES" if is_senior(age_range) else "NO"

            st.success(f"Gender: {gender}")
            st.success(f"Age Group: {age_range}")
            st.success(f"Senior Citizen: {senior_flag}")