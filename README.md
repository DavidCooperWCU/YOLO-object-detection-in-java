# YOLO-object-detection-in-java
This code is based on [the upstream fork](https://github.com/suddh123/YOLO-object-detection-in-java), A java based template for streaming video based object detection using only YOLO weights
and the [opencv c++ example for yolo](https://github.com/opencv/opencv/blob/aee828ac6ed3e45d7ca359d125349a570ca4e098/samples/dnn/yolo_detector.cpp).  



 ## What this project is about :

This fork is meant to make it easier to leverage YOLO using java and opencv. To that end, a gradle build file that uses opencv 4.9 from the org.openpnp implementation is used. This is currently configured to work out of the box on a mac with an ARM processor (M1-MX). However, you can extract the approprate .dll or .so file for other architectures from the jar that is built by gradle.





## Weights files

The onnx weights file was generated using the following python script:

```python
from ultralytics import YOLO
import cv2

# Load a COCO-pretrained YOLO8n model
model = YOLO("yolov8n.pt")
model.export(format='onnx') # Export to ONNX format

```

Note that you'll need python3, and you'll need to install ultralytics using pip3. (On a Mac it's much easier to do in a virtual environment using venv.)

If you want to modify this for use with other yolo models, then you'll want to look carefully at the [model specific code](https://github.com/opencv/opencv/blob/aee828ac6ed3e45d7ca359d125349a570ca4e098/samples/dnn/yolo_detector.cpp#L120-L139) of the opencv c++ example.
