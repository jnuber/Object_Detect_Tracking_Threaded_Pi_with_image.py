#!/usr/bin/env python3
# python  object_tracking_calibration_piCam.py

# Import the necessary packages

from __future__ import print_function
from imutils.video import FPS

########################################################################
# pi version added
from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray
from picamera import PiCamera
########################################################################

import imutils
import numpy as np
import cv2
import sys
import curses
from time import time
from time import sleep

homeFolder = "/home/pi/images"
calibration_image = "/home/pi/images/calibration_image_Pi.jpg"
print("Path:", homeFolder)

# from time import sleep
from subprocess import check_output

print("Version: 2017.05.08")
print(" Python: {}.{}.{}".format(sys.version_info[0],sys.version_info[1],sys.version_info[2]) )
print("  numpy: {}".format(np.__version__))
print("OpenCV2: {}\n".format(cv2.__version__))

########################################################################
# pi version added
# initialize the camera and grab a reference to the raw camera capture
camera = PiVideoStream().start()
# allow the camera to warmup
sleep(.5)
#########################################################################
camera.resolution = (120, 160) # Set Resoluton to 120 x 160 

status       = "Target Calibration Image - Press 'q' to exit"
instructions = "Set camera exactly 24inchs or 2ft from target."
while True:
	frame = camera.read()
	#cv2.putText(frame, status, (10,15), cv2.FONT_HERSHEY_SIMPLEX,.375,(000,000,255), 1)
	#cv2.putText(frame, instructions, (frame.shape[1]-frame.shape[1]+10, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX,.5,(000,255,000), 1)

	cv2.imshow("Target Calibration Image",frame)
	key = cv2.waitKey(1) & 0xFF
	sleep(.25)
	
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# Save calibration image -- OpenCV handles converting filetypes
frame = camera.read()
resize = imutils.resize(frame,height=240)
cv2.imwrite(calibration_image, resize)
print("")
print("   *************************************************")
print("Calibration Image Saved to: {}".format(calibration_image))
#print("Image Resolution: {}".format(frame[1],frame[0]))
camera.stop()
cv2.destroyAllWindows()

quit()
