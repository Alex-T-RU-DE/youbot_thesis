# Jetson Xavier software
## vision_youbot libraries
Custom libraries for OpenCV, PCL and utils are represnted in the ROS package vision_youbot.
## tkDNN
[TkDNN](https://github.com/ceccocats/tkDNN) platform is used as a base for YOLO detection. 
### demo.cpp
Main executable starting YOLO inference. This file was completely change to ensure compatibility with ROS. 
### DetectionNN.h
"void draw" is a main function, which draws the detected YOLO boxes. Based on this function, "drawYOLO" function is implemented. This function uses all features of vision_youbot package.
