# Liveness-Check-using-Blink-Detection-and-Edge-Detection
An implementation of face spoof detection using blink detection and edge detection.

This project is for the liveness detection of a Face using Webcam as well as a
pre-captured video of a person.

There are two phases to this project

External requirements to this project:
shape predictor file
test video file(preferably in mp4 format)

The first phase is to check using blink detection.
Over here the shape predictor dat file and the video file parameters should 
be entered into the code.

Then run the main.py file.

The method in which it works is by  taking the video in the form of a filestream
and checking for the facial landmarks; the eye landmark and then it calulates the eye ratio
for more than a certain threshold the subject face could be differentiated as live or fake.



EDGE DETCTION -  works by calculating the edge changes in the image array and then if the number
of changes in the array is more than a certain value the image would be declared fake.

Problem -  the main problem with it is that it requires the subject to be a plain background 
with almost no backgriund prop.


NOTE* -- This Blink detection will not work for video replay attack and will be a failure, mostly.Also
Edge detection would give false bifurcations in many cases.
For a perfect system architecture we can use depth detection for liveness detection.
