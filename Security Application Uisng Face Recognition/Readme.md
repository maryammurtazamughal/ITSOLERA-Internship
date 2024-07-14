
# Face Recognition Security Application
  This repository contains code for a Face Recognition Security Application built using Flask, OpenCV, MTCNN, and FaceNet.
 The application allows real-time face detection and recognition, comparing webcam input with stored face embeddings.

## Features:
  * Face Detection: Uses MTCNN for detecting faces in images and webcam feeds.
  * Face Recognition: Utilizes FaceNet for generating and comparing face embeddings.
  * Authorization: Determines if a detected face matches any stored face in the databases

## Usage:

### Uploading Images:

      Navigate to the homepage and upload an image file containing a face.
      The application will compare the uploaded face with stored faces in the database and determine authorization status.

### Using Webcam:

      Click the "Start Webcam" button on the homepage to access the webcam feature.
      Allow the browser to access your webcam.
      The application will capture your face and compare it with stored faces in the database in real-time.

## Prerequisites:
    * Python 3.x
    * MTCNN
    * FaceNet
    * OpenCV
    * Flask
