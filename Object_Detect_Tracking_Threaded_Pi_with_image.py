# USAGE
# python Object_Detect_Tracking_Threaded_Pi.py - Optimized
# by John Nuber
# Jan 2017
#
# Credits  
# Image processing is not a trivial task by any means. There's a level of
# complexity for the many different method in solving a image processing 
# problem. To that end, much... well a whole hell of a lot of research went
# into different aspects of object detection, determining object size, 
# distance to the camera, applied methods for maximum performance to name 
# a few. If you share this code, with others, its imperative credit is given
# to the following:
#   Adrian Rosebrock Ph.D - http://www.pyimagesearch.com/
#   Dave Jones - Author of piCamera Package - https://picamera.readthedocs.io/en/release-1.13/
#   Satya Mallick Ph.D - https://www.learnopencv.com/
#   Raspberry Pi Foundation - https://www.raspberrypi.org/learning/getting-started-with-picamera/
#
#=======================================================================
# Implementation Overview and Considerations
#
# This object tracking solutions utilize the "Triangle Similarity" method.
# In brief, the triangle similarity takes an object (marker) with a known
# width. The object is placed at some distance from the camera. Preferably
# the same camera to be used for detection and tracking.
#
# A photo/image is taken of the object using the same camera to be used for 
# object detection. We then measure the apparent width in pixels. Using 
# this image as a fixed reference point, if the camera moves away from the
# object, the number of pixels measuring the object's width decreases. If 
# the camera moves closer to the object, the number of pixels increases. 
# One can calculate the distance with pretty good accuracy. The higher the
# quality of the reference image, the better the accuracy. These attributes
# include: good lighting, good color contrast, accurate distance, as close to
# 90% camera angle as possible. In some cases, camera calibration and 
# focal length maybe required. However I found using the piCam,pinhole or
# fish-eye distortion wasn't an issue.
#  
# Solution Approach
# This solution uses a reference or calibration image of the object to 
# track. The object/marker width is determined by the number of pixels.
# Once this has been established, the main loop does the following: 
#   - looks for an object with similar contours as the reference object
#   - if found, target identified
#   - determines target width in pixels
#   - calculates distance based on a % difference from the
#     reference/calibration image
#   - locate target center
#   - display/print target details 
#   - for Raspberry Pi, get CPU temperature as well.
#
# This version previews the target, paints the target's boarders and provides
# target data on the preview screen. This an excellent method of viewing
# and debugging the code. Also included were performance stats for the 
# various functions. If using for robotics or autonomous mode, all the 
# above can be commented/removed for maximum performance. Through testing,
# with good target LIGHTING, I was achieving 30+ FPS. (Raspberry Pi 3, 
# piCam ver2, Multitheaded Camera streaming feed, Python 3.6.
# 
# NOTE: This program uses the OpenCV cv2.waitKey() for interactive
# termination. cv2.waitKey() only works with an active cv2 window panel. If you
# comment out all unnecessary visual previews, you need at lest ONE image
# window open to terminate a running version. Consider having the 
# cv2.imshow("Calibration Image",image) or the cv2.namedWindow() and 
# cv2.moveWindow() feature enabled. This will allow you to comment out
# none essential code and still run and terminate the program.
#
#=======================================================================
# import packages
from __future__ import print_function
from imutils.video import FPS

########################################################################
# pi version added
# *** NOTE: The PiVideoStream was modified to purposely set the camera
#           resolution to 320x240 and black and white mode.  Doing so
#           further improved perform by not having to resize the image 
#           stream and convert it to a gray scale image required for 
#           the Cannying (line detection) process. These settings are
#           only used in the Threaded mode.
from imutils.video.pivideostream import PiVideoStream
# from picamera.array import PiRGBArray # This loads the video into numpy array format.
# from picamera import PiCamera
########################################################################

# import imutils
import numpy as np
import cv2
import sys
# import curses
from time import time
from time import sleep
from subprocess import check_output

print("Version:2017.03.15")
print(" Python: {}.{}.{}".format(sys.version_info[0],sys.version_info[1],sys.version_info[2]) )
print("  numpy: {}".format(np.__version__))
print("OpenCV2: {}\n".format(cv2.__version__))


def auto_canny(image, sigma=0.33):    # lower % = tighter canny. 33% best for most cases.
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0,  (1.0 - sigma) * v))
	upper = int(min(255,(1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	
	return edged

def find_marker(image):
	(_, cnts, _) =cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	if not cnts:
		return 0
	
	c = max(cnts, key = cv2.contourArea)
	return cv2.minAreaRect(c)           # compute the bounding box of the of the paper
								 		 # region and return it's value
									 		
def distance_to_camera(knownWidth, focalLength, perWidth):
	if perWidth == 0:
		return 1
	# compute and return the distance from the object to the camera. 
	return (knownWidth * focalLength) / perWidth


#=========================================================================================
#=========================================================================================
# set calibration image width and distance from camera in inches

KNOWN_DISTANCE =  24     # inches
KNOWN_WIDTH    =  3      # inches

# load calibration image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the marker in the image, and initialize
# the focal length. If you used the Calibration Utility to help capture the
# calibration image, the piCamera capture the image at a 320x240 resolution.
# it CRITICAL for simplicity and accuracy, and performance, to match the calibration
# image resolution to the video stream resolution or visa-versa.

print("Loading Calibration Image....")
image  = cv2.imread("/home/pi/images/calibration_image.jpg")

gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
edged   = auto_canny(gray)

# The following was used for troubleshooting
# cv2.imshow("Calibration Image",image)
# cv2.imshow("Cal Image Edged", edged)

# Marker = the number of pixels that equate to the width
marker = find_marker(edged)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
calPix = int(marker[1][0])
inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

print("Calibration Image Height:", image.shape[0])
print("Calibration Image Width: ", image.shape[1])
print("Cal Image Pixels Width:  ", calPix)
print("Focal Length: {:4.2f}".format(focalLength))
print("Distance: {:4.2f} inches".format(inches))
print("\nInitalizing camera in multi-thread mode")
print("Initalizing FPS stats collector in multi-thread mode\n")


# The following were used to performance statistics collection
# comment or remove them when you no longer need them.
capture_time      = []
GBEM_time         = []
distance_time     = []
threshold_time    = []
findContours_time = []
drawContours_time = []
crosshairs_time   = []
show_time         = []
exitKey_time      = []
loop_time         = []

fps = FPS().start()

########################################################################
# pi version added
# initialize the camera and grab a reference to the raw camera capture
# Then allow the camera to warm up
# Any delays less then 1 second I found to be a problem.
camera = PiVideoStream().start()
sleep(2)
#########################################################################

frame = camera.read()
f_h = (frame.shape[0])
f_w = (frame.shape[1])
print("HxW {}x{}".format(int(f_h),int(f_w)))

# ======================================================================
#    Main Loop
# ======================================================================

cv2.namedWindow("Target")               # Reserve or name a window for use later
cv2.moveWindow( "Target", 800,100)      # Position window on user display at x,y coordinates

while True:
	start= time()
	loop_start= start
	frame= camera.read()
	f_h = (frame.shape[0])/2
	f_w = (frame.shape[1])/2

	status = "No Targets"
	capture_time.append(time() - start) 
	
	start = time()
	# convert the frame to gray scale, blur it, and detect edges
	edged =auto_canny(frame)
	# cv2.imshow("Edged",edged)         # Show Edged Lines - used for troubleshooting
	marker = find_marker(edged)         # Detect the lines in the frame
	GBEM_time.append(time() - start)    #Gray, Blur, Edge, Marker time

	start = time()
	if marker:
		# Calculate distance and find contours in the edge map
		# inches = KNOWN_DISTANCE  * (calPix/(marker[1][0]))
		inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
		(_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
		distance_time.append(time() - start)
		
		# loop over the contours
		start = time()
		for c in cnts:
		# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.01 * peri, True)       # ??

		# ensure that the approximated contour is "roughly" rectangular
			if len(approx) >= 4 and len(approx) <= 6:
				# compute the bounding box of the approximated contour and
				# use the bounding box to compute the aspect ratio
				(x, y, w, h) = cv2.boundingRect(approx)
				aspectRatio = w / float(h)

				# compute the solidity of the original contour
				area = cv2.contourArea(c)
				hullArea = cv2.contourArea(cv2.convexHull(c))
				solidity = area / float(hullArea)
			
				# compute whether or not the width and height, solidity, and
				# aspect ratio of the contour falls within appropriate bounds
				keepDims = w > 25 and h > 25
				keepSolidity = solidity > 0.9
				# keepAspectRatio = aspectRatio >= 0.95 and aspectRatio <= 1.05  # is a square
				# keepAspectRatio = aspectRatio >= 0.5  and aspectRatio <= 1.5

				# ensure that the contour passes all our tests
				# if keepDims and keepSolidity and keepAspectRatio:
				findContours_time.append(time() - start)
				if keepDims and keepSolidity:
					start = time()
					# draw an outline around the target and update the status text
					cv2.drawContours(frame, [approx], -1, (0, 0, 255), 1)
					status = "Target(s) Acquired"
					drawContours_time.append(time() - start)
					
					start = time()
					# compute the center of the contour region and draw the  crosshairs
					M = cv2.moments(approx)
					(cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
					(startX, endX) = (int(cX - (w * 0.1)), int(cX + (w * 0.1)))
					(startY, endY) = (int(cY - (h * 0.1)), int(cY + (h * 0.1)))
					cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 2)
					cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 2)
					
					cv2.putText(frame, "%.2fft" % (inches/12), (frame.shape[1] - 50, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX,.5,(000,255,000), 1)
					cv2.putText(frame,"x:{} y:{}".format(cX-f_w , cY-f_h), (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5,(000,255,000),1)
					crosshairs_time.append(time()- start)
				
		start = time()		
		# draw the status text on the frame
		cv2.putText(frame, status, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
		cpuTemp = check_output(["vcgencmd", "measure_temp"]).decode("UTF-8")
		# cv2.putText(frame, cpuTemp, (frame.shape[1] - 300, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX,.5,(000,255,000), 1)	

		# show the frame and record if a key is pressed
		cv2.imshow("Target", frame)
		show_time.append(time() - start)
		
		start = time()
		fps.update()

	key = cv2.waitKey(1) & 0xFF
	
	if not status == "No Targets":
		# print("{} x:{} y:{} ft:{:.2f}".format(status,(cX-f_w),(cY-f_h),(inches*3)/12) )
		print("{} P:{} x:{} y:{} ft:{:.2f} {} ".format(status,int(marker[1][0]),(cX-f_w),(cY-f_h),(inches)/12, cpuTemp) )
	
	# if the 'q' key is pressed, stop the loop
	if (key == ord("q") or key == ord("Q") or key == chr(27)):
		break
	exitKey_time.append(time() - start)
	loop_time.append(time() - loop_start)
	
# cleanup the camera and close any open windows
fps.update()
fps.stop()
camera.stop()
cv2.destroyAllWindows()
print("============================================")
print("\tElasped time: {:.2f}".format(fps.elapsed()))
print("\tApprox. FPS: {:.2f}".format(fps.fps()))

# ======================================================================
#    END Loop
# ======================================================================

def mean(l):
	#print('Sum {} : Len {}'.format(sum(l), len(l) ) )
	return sum(l) / len(l)

mean_capture_time      = mean(capture_time)
mean_GBEM_time         = mean(GBEM_time)
mean_distance_time     = mean(distance_time)
mean_findContours_time = mean(findContours_time)
mean_drawContours_time = mean(drawContours_time)
mean_crosshairs_time   = mean(crosshairs_time)
mean_show_time         = mean(show_time)
mean_exitKey_time      = mean(exitKey_time)
mean_loop_time         = mean(loop_time)

print('\n    Average loop time: %.3fs (%.2ffps)'% (mean_loop_time,        (1/mean_loop_time)))
print('      Capture loop time: %.3fs (%.1f%% )'% (mean_capture_time,     (mean_capture_time      * 100/ mean_loop_time)))
print('         GBEM loop time: %.3fs (%.1f%% )'% (mean_GBEM_time,        (mean_GBEM_time         * 100/ mean_loop_time)))
print('    Calc Dist loop time: %.3fs (%.1f%% )'% (mean_distance_time,    (mean_distance_time     * 100/ mean_loop_time)))
print('Find Contours loop time: %.3fs (%.1f%% )'% (mean_findContours_time,(mean_findContours_time * 100/ mean_loop_time)))
print('Draw contours Loop time: %.3fs (%.1f%% )'% (mean_drawContours_time,(mean_drawContours_time * 100/ mean_loop_time)))
print('   Crosshairs loop time: %.3fs (%.1f%% )'% (mean_crosshairs_time,  (mean_crosshairs_time   * 100/ mean_loop_time)))
print('  Show Frames loop time: %.3fs (%.1f%% )'% (mean_show_time,        (mean_show_time         * 100/ mean_loop_time)))

sleep(3)
