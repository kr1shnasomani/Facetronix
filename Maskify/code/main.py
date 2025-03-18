# System and logging configuration
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import required libraries
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Path to images and model files
image_path = r"C:\Users\krish\OneDrive\Desktop\image.jpg"
result_path = r"C:\Users\krish\OneDrive\Desktop\output.jpg"
prototxt = r"C:\Users\krish\Downloads\deploy.prototxt"
caffemodel = r"C:\Users\krish\Downloads\res10_300x300_ssd_iter_140000.caffemodel"
h5 = r"C:\Users\krish\Downloads\model.h5"

# Detect faces and predict mask
def detect_and_predict_mask(image, faceNet, maskNet, confidence_threshold=0.2):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            if face.size == 0: continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Load models
faceNet = cv2.dnn.readNet(prototxt, caffemodel)
maskNet = load_model(h5)

# Load input image
image = cv2.imread(image_path)

# Perform face mask detection
(locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)

# Annotate image with results
for (box, pred) in zip(locs, preds):
    (startX, startY, endX, endY) = box
    (mask, withoutMask) = pred

    label = "Mask" if mask > withoutMask else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

    font_scale = max(0.8, (endX - startX) / 200)
    thickness = int(font_scale * 2)
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y_offset = max(startY - 10, text_height + 10)

    # Draw background rectangle for text
    cv2.rectangle(image, (startX, y_offset - text_height - 10), (startX + text_width, y_offset + 10), color, -1)
    
    # Draw label text
    cv2.putText(image, label_text, (startX, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Draw bounding box
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Save output image
cv2.imwrite(result_path, image)
print(f"Output saved to {result_path}")