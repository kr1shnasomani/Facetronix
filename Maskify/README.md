<h1 align="center">Maskify</h1>
This script detects faces in an image using a Caffe model and predicts mask usage with a Keras classifier. It annotates faces with bounding boxes and labels ("Mask" or "No Mask") based on predictions, saving the annotated image to a specified path.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install tensorflow opencv-python numpy
   ```

2. Download the following models:

   a. [deploy.prototxt](https://github.com/kr1shnasomani/Maskify/blob/main/model/deploy.prototxt)

   b. [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/kr1shnasomani/Maskify/blob/main/model/res10_300x300_ssd_iter_140000.caffemodel)

   c. [model.h5](https://github.com/kr1shnasomani/Maskify/blob/main/model/model.h5)

3. Enter the path of the image and the location where you want to save the output.

4. Upon running the code, it will display the results in the specified path.

## Result:

  Input Image:

  ![image](https://github.com/user-attachments/assets/f100ab3a-2694-450f-9308-ca42d17223a2)

  Output Image:

  ![output](https://github.com/user-attachments/assets/8fba4ee2-8bc0-41cc-81e2-a48b86153c8c)

## Overview:
This Python script implements a face mask detection system using deep learning and OpenCV. Below is an organized explanation of the components and workflow:

### **1. Libraries and Environment Configuration:**
- **TensorFlow Logging:** Suppresses TensorFlow debug and error messages for cleaner output.
- **Required Libraries:** Imports libraries for image processing (`OpenCV`), numerical computations (`NumPy`), and deep learning (`TensorFlow` and Keras).

### **2. Paths to Required Files:**
- `image_path`: Input image location.
- `result_path`: Output image save location.
- `prototxt`: Path to the Caffe model architecture for face detection.
- `caffemodel`: Path to the pre-trained weights for face detection.
- `h5`: Path to the Keras model for predicting whether a face has a mask or not.

### **3. `detect_and_predict_mask` Function:**
- **Purpose:** Detects faces in an image and predicts if each face is wearing a mask.
- **Inputs:**
  - `image`: The input image.
  - `faceNet`: The pre-trained face detection model (Caffe).
  - `maskNet`: The mask classification model (Keras).
  - `confidence_threshold`: Minimum confidence to consider a face detection as valid.
- **Workflow:**
  1. Preprocesses the image into a blob format suitable for the face detection model.
  2. Passes the blob to the face detection model to get bounding box predictions.
  3. Filters out low-confidence detections.
  4. For each detected face:
     - Extracts the region of interest.
     - Preprocesses the face for input into the mask classifier.
  5. Uses the mask classifier to predict whether the face has a mask.
- **Outputs:**
  - `locs`: List of bounding box coordinates for detected faces.
  - `preds`: List of predictions (probabilities for "Mask" and "No Mask").

### **4. Loading Models:**
- **Face Detection Model:** A pre-trained Caffe model (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`).
- **Mask Detection Model:** A Keras model (`model.h5`) that classifies faces as "Mask" or "No Mask."

### **5. Face Mask Detection Process:**
1. **Image Loading:** Reads the input image using OpenCV.
2. **Face Detection and Prediction:**
   - Calls `detect_and_predict_mask` to locate faces and predict mask usage.
3. **Annotating Results:**
   - Iterates through detected faces (`locs`) and corresponding predictions (`preds`).
   - Draws bounding boxes around faces.
   - Annotates each box with:
     - Label: "Mask" or "No Mask."
     - Confidence: Probability of the prediction.
   - Uses color coding:
     - Green for "Mask."
     - Red for "No Mask."

### **6. Saving the Annotated Image:** 
Saves the annotated image to the specified location (`result_path`) and prints a success message.

### **Key Features**
- **Dynamic Font Scaling:** Adjusts label font size and thickness based on bounding box size for better readability.
- **Real-Time Adaptability:** Although this script processes a single image, it can be adapted for video or real-time applications with minor modifications.
