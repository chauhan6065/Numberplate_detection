import streamlit as st
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import re
from PIL import Image

st.title("🚗 License Plate Recognition App")

# Load YOLO model
model = YOLO("best.pt")  # Place your trained weights in the same folder
reader = easyocr.Reader(['en'])

uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # YOLO detection
    results = model.predict(image, conf=0.25)

    if len(results[0].boxes.xyxy) == 0:
        st.warning("⚠️ No license plate detected!")
    else:
        # Copy image for annotation
        annotated_img = image.copy()
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop plate and OCR
            plate_img = image[y1:y2, x1:x2]
            ocr_result = reader.readtext(plate_img)
            texts = [res[1] for res in ocr_result if res[2] > 0.3]
            plate_text = "".join(texts)
            clean_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
            st.success(f"Detected License Plate: {clean_text if clean_text else 'Not readable'}")

        # Show annotated image
        st.image(annotated_img, caption="Detected Plate(s)")
