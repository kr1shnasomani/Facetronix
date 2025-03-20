<h1 align="center">Facetronix</h1>
<p align="center" style="margin-top:30px;">
  <img src="https://github.com/user-attachments/assets/7f530546-5482-47db-9769-4a99878fe00c" height="300cm"/>
</p>

## Project Index:
| Name | Description |
|------|-------------|
| [FaceCapture](https://github.com/kr1shnasomani/Facetronix/tree/main/FaceCapture) | This script detects faces in an image using a Haar cascade classifier, draws bounding boxes around detected faces, saves the result, and generates a grid of cropped, resized faces using Matplotlib. It handles multiple faces and saves outputs to specified paths. |
| [FaceFeel](https://github.com/kr1shnasomani/Facetronix/tree/main/FaceFeel) | This project involves building a Convolutional Neural Network (CNN) for facial emotion recognition using the FER-2013 dataset. It preprocesses images, trains the model, evaluates accuracy, visualizes performance and includes a prediction function to classify emotions from images with confidence scores. |
| [Maskify](https://github.com/kr1shnasomani/Facetronix/tree/main/Maskify) | This script detects faces in an image using a Caffe model and predicts mask usage with a Keras classifier. It annotates faces with bounding boxes and labels ("Mask" or "No Mask") based on predictions, saving the annotated image to a specified path. |

## Repository Structure:
```
Facetronix/
├── FaceCapture/
│   ├── code/
│   │   └── main.py
│   ├── dataset/
│   │   └── image.jpg
│   ├── result/
│   │   ├── boundingbox.jpg
│   │   └── croppedface.jpg
│   └── README.md
├── FaceFeel/
│   ├── code/
│   │   └── main.ipynb
│   ├── model/
│   │   └── model.keras
│   └── README.md
├── Maskify/
│   ├── code/
│   │   └── main.py
│   ├── dataset/
│   │   └── image.jpg
│   ├── model/
│   │   ├── deploy.prototxt
│   │   ├── model.h5
│   │   └── res10_300x300_ssd_iter_140000.caffemodel
│   ├── result/
│   │   └── output.jpg
│   └── README.md
└── README.md
```
