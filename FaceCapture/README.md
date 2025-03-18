<h1 align="center">FaceCapture</h1>
This script detects faces in an image using a Haar cascade classifier, draws bounding boxes around detected faces, saves the result, and generates a grid of cropped, resized faces using Matplotlib. It handles multiple faces and saves outputs to specified paths.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install opencv-python matplotlib
   ```
   
2. Enter the path of the input image and the location where you want to save the output

3. Upon running the code it will detect the faces and save two files - `boundingbox.jpg` and `croppedface.jpg`

## Result:

   Input Image:

   ![image](https://github.com/user-attachments/assets/edf8187e-7ef7-4382-865d-3148a48a135d)

   Output Image:

   a. `boundingbox.jpg`

   ![boundingbox](https://github.com/user-attachments/assets/b84fdd1b-c7ce-485c-abfd-578a029ea6c2)

   b. `croppedface.jpg`

   ![croppedface](https://github.com/user-attachments/assets/99dd9dbb-b3a7-4646-b458-ce16743df22d)

## Overview:
This code demonstrates a face detection and visualization pipeline using OpenCV's Haar Cascade Classifier. Below is a structured overview:

1. **Library Imports**:
   - cv2: For image processing and face detection.
   - matplotlib.pyplot: To visualize and save the cropped faces in a grid layout.
   - math: For calculating rows needed to organize cropped faces.

2. **Paths**:
   - image_path: Input image for face detection.
   - boundingbox_path: Output image path for saving the image with bounding boxes.
   - croppedface_path: Output image path for saving a grid of cropped faces.

3. **Utility Functions**:
   - **convertToRGB**: Converts images from BGR (OpenCV default) to RGB format for proper visualization with Matplotlib.
   - **detect_faces**: 
     - Takes a Haar cascade, an image, and a scale factor for face detection.
     - Converts the image to grayscale for better detection performance.
     - Detects faces using detectMultiScale.
     - Draws bounding boxes on the detected faces and extracts cropped face regions.

4. **Pipeline**:
   - **Load Image**: Read the input image using cv2.imread.
   - **Load Haar Classifier**: Load OpenCV's pre-trained Haar Cascade for face detection.
   - **Detect Faces**: Detects faces, returns cropped face regions, bounding boxes, and an image with bounding boxes drawn.
   - **Save Bounding Box Image**: The processed image with bounding boxes is saved to the specified path.

5. **Visualize and Save Cropped Faces**:
   - If faces are detected:
     - Resize cropped faces to 100x100 pixels for uniformity.
     - Display faces in a grid (5 faces per row, dynamically calculated rows).
     - Save the grid as an image.
   - If no faces are detected, a message is printed.

6. **Output Paths**: Logs the paths of the saved bounding box image and cropped face grid.
