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

![image](https://github.com/user-attachments/assets/32836ac2-e2cb-4faa-a6e2-6b1c5b8a3376)x

## Overview:
