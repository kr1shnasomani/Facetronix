# Import the required libraries
import cv2
import matplotlib.pyplot as plt
import math

# Paths for input and output images
image_path = r'C:\Users\krish\OneDrive\Desktop\image.jpg'
boundingbox_path = r'C:\Users\krish\OneDrive\Desktop\boundingbox.jpg'
croppedface_path = r'C:\Users\krish\OneDrive\Desktop\croppedface.jpg'

# Convert BGR image to RGB
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces and return cropped faces
def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = colored_img[y:y+h, x:x+w]
        cropped_faces.append(face)
    
    return cropped_faces, faces, img_copy

# Load image
test_img = cv2.imread(image_path)

# Load Haar classifier
haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Detect faces
cropped_faces, faces_detected, img_with_faces = detect_faces(haar_face_cascade, test_img)

# Save image with bounding boxes
cv2.imwrite(boundingbox_path, img_with_faces)

# Save detected faces in subplots with 5 faces per row
if len(cropped_faces) > 0:
    num_faces = len(cropped_faces)
    num_rows = math.ceil(num_faces / 5)
    
    resized_faces = [cv2.resize(face, (100, 100)) for face in cropped_faces]
    fig, axes = plt.subplots(num_rows, 5, figsize=(15, num_rows * 3))
    
    axes = axes.flatten()
    for i, face in enumerate(resized_faces):
        axes[i].imshow(convertToRGB(face))
        axes[i].axis('off')
        axes[i].set_title(f"Face {i+1}")
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Save the cropped faces image
    plt.tight_layout(pad=1.0) 
    plt.savefig(croppedface_path)
else:
    print("No faces detected.")

# Output paths
print(f"Bounding Box Image Saved to: {boundingbox_path}")
print(f"Cropped Faces Image Saved to: {croppedface_path}")