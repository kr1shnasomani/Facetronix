<h1 align="center">FaceCapture</h1>
This project implements an age and gender prediction system using OpenCV's deep learning module. It detects faces in an image, identifies gender as 'Male' or 'Female,' and estimates the age range. The model employs pre-trained DNN models for accurate predictions and overlays results directly on the image.

## Execution Guide:
1. Run the following command in the terminal:
   ```
   pip install opencv-python protobuf
   ```

2. Download the models using the following links:
 
   a. [age_deploy.prototxt](https://github.com/kr1shnasomani/Facetronix/blob/main/GenAgeNet/model/age_deploy.prototxt)
   
   b. [age_net.caffemodel](https://github.com/kr1shnasomani/Facetronix/blob/main/GenAgeNet/model/age_net.caffemodel)
   
   c. [gender_deploy.prototxt](https://github.com/kr1shnasomani/Facetronix/blob/main/GenAgeNet/model/gender_deploy.prototxt)
   
   d. [gender_net.caffemodel](https://github.com/kr1shnasomani/Facetronix/blob/main/GenAgeNet/model/gender_net.caffemodel)
   
   e. [opencv_face_detector.pbtxt](https://github.com/kr1shnasomani/Facetronix/blob/main/GenAgeNet/model/opencv_face_detector.pbtxt)
   
   f. [opencv_face_detector_uint8.pb](https://github.com/kr1shnasomani/Facetronix/blob/main/GenAgeNet/model/opencv_face_detector_uint8.pb)
   
## Result:

Input:

![image](https://github.com/user-attachments/assets/fbb0b199-7489-4a1b-8c77-01d45de4ad8b)

Output:

![image](https://github.com/user-attachments/assets/32836ac2-e2cb-4faa-a6e2-6b1c5b8a3376)

## Overview:
The given code is a **Gender and Age Detection System** that uses **OpenCV's Deep Neural Network (DNN) module** to identify faces in an image and predict their gender and age.  The following is the detailed explaination of the code:

### **1. Import Required Libraries**  
- `cv2` (OpenCV) → For image processing and deep learning inference  
- `math` → (Unused in the script)  
- `os` → For file path management  

### **2. Define `highlightFace()` Function**  
- Loads an **OpenCV DNN face detection model**.  
- Uses `cv2.dnn.blobFromImage()` to preprocess the image.  
- Passes the processed image into the model to detect faces.  
- Draws **green bounding boxes** around detected faces.  
- Returns the image with detected faces and their bounding box coordinates.  

### **3. Load Pre-trained Models**  
The code loads deep learning models trained on the **Caffe framework** to detect **faces, gender, and age**.  
- **Face detection model**  
  - `opencv_face_detector.pbtxt` (Model configuration)  
  - `opencv_face_detector_uint8.pb` (Pre-trained model)  
- **Age detection model**  
  - `age_deploy.prototxt` (Configuration)  
  - `age_net.caffemodel` (Pre-trained model)  
- **Gender detection model**  
  - `gender_deploy.prototxt` (Configuration)  
  - `gender_net.caffemodel` (Pre-trained model)  

### **4. Process the Image**  
- Reads an image (`dataset/image.jpg`) from the dataset.  
- Passes it through the `highlightFace()` function to detect faces.  
- If no faces are detected, it prints `"No face detected"` and exits.  
- Otherwise, for each detected face:  
  1. Extracts the face region with a **padding of 20 pixels**.  
  2. Converts the face into a **blob** and passes it through:  
     - **Gender detection model** → Predicts Male/Female  
     - **Age detection model** → Predicts an age range  
  3. Displays the detected gender and age above the face using `cv2.putText()`.  

### **5. Save and Output Results**  
- Saves the processed image with the detected face(s), gender, and age to `result/output.jpg`.  
- The extracted information is also printed in the console.  
