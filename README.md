# Vehicle-Detection-OpenCV

## Vehicle Detection
# My model and approach
We first need to read the video, capture it and create a mask for it.
Creating this mask will help us to analyse the frames of the video better.
We will use createBackgroundSubtractorMOG2 for creating the background mask for the video frames. We will also enter threshold and history of frames to be compared for object detection.
The history of frames and the varThreshold would change wrt the video under preview.
The frames change constantly, so we need to narrow our detection strategy by fine tuning the frame changes that we consider as legitimate object moves. Track that object and be stuch to that.
Next we will draw contours over the objects for better counting.
We would also have to set a target area of detection. 
This target area would be 250 above and below the line, with 50 forward and backward along the length of the line.
Next we will set the contour area that has to be registered as a valid object (in this case a vehicle).
We then construct bounding rectangles around the contours to display the object being tracked.
These contours have been embedded with a centre circle. Now wheneer the centre of the circle passes the input line, the vehicle gets counted. 

# Reason for choosing this model
This works in the night mode video also. This can be fine tuned as per the video and can be made perfect. Like if we know a particular camera's angle, the script can pe perfectly altered and made as per the frames in the video.

# Other Models I could choose from
YOLOv3 and YOLOv2 are the other choices I could go for, but their functioning and development is cumbersome, and there arent pefect libraries that match all the dependencies of other packages in use. I still have attached that code too, but there are several libraries that work with python2 in the algorithm's backend, that the code just doesnt compile. Building a virtual environment and installing specific versions of the packages also didnt help, as there are some issues related to python2, numpy, keras and the allied libraries that have been solved in python3, but the libraries do no tsupport python3.

# Current Model Correctness
This model is robust and can deliver accuracy if the the script is finetuned as per the video and the code is fed positive and negative training and testing samples to biuld accuracy.

# Video Output
The outut opens in a window via cv2.imshow().

