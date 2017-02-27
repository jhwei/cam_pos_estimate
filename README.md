# Camera Position Estimation

Author: Jiahui Wei

This is the Python code solution to a DroneDeploy coding challenge.

## Challenge Statement

The images are taken from different positions and orientations with an iPhone 6. Each image is the view of a pattern on a flat surface. The original pattern that was photographed is 8.8cm x 8.8cm and is included in the zip file. Write a Python program that will visualize (i.e. generate a graphic) where the camera was when each image was taken and how it was posed, relative to the pattern.

You can assume that the pattern is at (0,0,0) in some global coordinate system and are thus looking for the x, y, z and yaw, pitch, roll of the camera that took each image.

## Library for Project

OpenCV
Python package: numpy, math, cv2 (to use OpenCV), matplotlib 

## Output

Output is in the *output* folder.

Output graphic image is in *cam_pos.jpg*. Numerical output is in *stat.txt*.
