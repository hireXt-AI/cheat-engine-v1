# Face Detection & Interview Monitoring System

## A Python-based real-time face detection and monitoring system built with OpenCV and Haar Cascades. The system supports face detection, reference-image matching, eye-state monitoring, lighting detection, and eyeball/screen-gaze tracking.

Features
Face detection using OpenCV Haar Cascades
Reference-image matching with live camera feed
Person presence detection
Low-light detection
Light ON/OFF detection
Eye-state detection:
Eyes open
Eyes closed/occluded
Sunglasses detected
Eyeball movement and screen-gaze tracking
Annotated output for static images
Configurable face-matching parameters
Installation

Install the required dependencies:

pip install -r requirements.txt
Usage
1. Detect Faces in an Image

```
python face_detection.py --image path/to/your_image.jpg
```

The system will detect faces and save the annotated image as:

detected_faces.jpg
2. Live Camera with Reference Image

To compare the person in front of the camera with a reference image:

python face_detection.py --reference path/to/reference_image.jpg --camera

The system will open the webcam and perform real-time monitoring.

3. Live Camera Detection

Camera mode provides the following detections:

Person present/not present
Face detection
Reference-image matching
Low-light detection
Light status
Eye state
Sunglasses detection
Eyeball movement
Screen-gaze tracking
4. Customize Output File

You can specify a custom output filename for static image detection:

```
python face_detection.py --image path/to/your_image.jpg --output output_file.jpg
```
Face Matching Parameters

The matching behavior can be customized using:

Camera Index

Select a specific camera device:

--camera-index 0
Match Threshold

Controls how strict the reference-image matching is.

--match-threshold 0.25

Lower values generally make matching easier, while higher values make matching stricter.

Minimum Good Matches

Specifies the minimum number of ORB feature matches required for a successful match:

--min-good-matches 12
Example
python face_detection.py \
    --reference path/to/reference_image.jpg \
    --camera \
    --match-threshold 0.25 \
    --min-good-matches 12
Demo Script

A simplified demo script is also available for launching the live monitoring workflow:

python camera_demo.py path/to/reference_image.jpg

This starts the live camera workflow using the provided reference image.

Project Structure
cheat-engine-v1/
│
├── face_detection.py
├── camera_demo.py
├── requirements.txt
├── README.md
└── ...
Requirements
Python 3.x
OpenCV
NumPy
Required packages listed in requirements.txt
Notes

For best face-matching results:

Use a clear reference image.
Keep the face clearly visible.
Ensure sufficient lighting.
Avoid excessive face occlusion.
Use a suitable camera resolution.