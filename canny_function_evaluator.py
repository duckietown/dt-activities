#!/usr/bin/env python

"""
TODO List all the steps required to create a perfect Canny detector.
In this way we can create functions with different performances
removing some features.
"""

import cv2
import numpy as np			
from matplotlib import pyplot as plt
from MOOCfunctions.core import CVMoocFunctions

def main():
	mooc = CVMoocFunctions(distortion=False)

	image_original = mooc.getImage()

	lowThreshold = 75
	highThreshold = 150

	image_blurred = cv2.blur(image_original, (3, 3))
	image_canny_cv2 = cv2.Canny(image_blurred, lowThreshold, highThreshold)

	image_canny_our = ourCanny(image_original, lowThreshold, highThreshold)

	#Calculate errors
	mooc.calculateErrors(image_canny_cv2, image_canny_our)
	
	# Plot images side by side
	plotImages(image_original, image_canny_cv2, image_canny_our)
	

def plotImages(image_original, image_canny_cv2, image_canny_our):
	plt.subplot(131),plt.imshow(image_original,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(132),plt.imshow(image_canny_cv2,cmap = 'gray')
	plt.title('Cv2 Canny'), plt.xticks([]), plt.yticks([])
	plt.subplot(133),plt.imshow(image_canny_our,cmap = 'gray')
	plt.title('Our Canny'), plt.xticks([]), plt.yticks([])

	plt.show()

def ourCanny(input_image, minVal, maxVal):
	# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
	# 1) Convert image to greyscale image
	if (input_image.shape[2] > 1):
		input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

	# 2) Blur image
	image_blurred = cv2.blur(input_image, (3, 3))

	# 3) Gradient Image
	Ix = cv2.Sobel(image_blurred, cv2.CV_64F, 1, 0, ksize=3)
	Iy = cv2.Sobel(image_blurred, cv2.CV_64F, 0, 1, ksize=3)
	# Calculate magnitude as hypotenuse of triangle formed by Ix and Iy
	gradientMagnitude = np.hypot(Ix, Iy)
	# Normalize gradient magnitude for display purposes
	gradientMagnitude = (gradientMagnitude / gradientMagnitude.max()) * 255

	gradientDirection = np.arctan2(Iy, Ix)

	# 4) Non-maximum suppression
	rows, columns = gradientMagnitude.shape
	gradientNonMaxSuppression = np.zeros((rows, columns), dtype=np.int32)
	gradientAngle = (gradientDirection * 180.0) / np.pi
	# Just need direction
	gradientAngle[gradientAngle < 0] += 180

	for i in range(1, rows - 1):
		for j in range(1, columns - 1):
			try:
				q = 255
				r = 255
				# angle 0
				if (0 <= gradientAngle[i, j] < 22.5) or (157.5 <= gradientAngle[i, j] <= 180):
					q = gradientMagnitude[i, j + 1]
					r = gradientMagnitude[i, j - 1]
				# angle 45
				elif (22.5 <= gradientAngle[i, j] < 67.5):
					q = gradientMagnitude[i + 1, j - 1]
					r = gradientMagnitude[i - 1, j + 1]
				# angle 90
				elif (67.5 <= gradientAngle[i, j] < 112.5):
					q = gradientMagnitude[i + 1, j]
					r = gradientMagnitude[i - 1, j]
				# angle 135
				elif (112.5 <= gradientAngle[i, j] < 157.5):
					q = gradientMagnitude[i - 1, j - 1]
					r = gradientMagnitude[i + 1, j + 1]

				if (gradientMagnitude[i, j] >= q) and (gradientMagnitude[i, j] >= r):
					gradientNonMaxSuppression[i, j] = gradientMagnitude[i, j]
				else:
					gradientNonMaxSuppression[i, j] = 0

			except IndexError as e:
				pass

	# 5) Double threshold
	gradientDoubleThreshold = np.zeros((rows, columns), dtype=np.int32)

	weak = np.int32(255)
	strong = np.int32(255)

	gradientDoubleThreshold[gradientNonMaxSuppression >= minVal / 3] = weak
	gradientDoubleThreshold[gradientNonMaxSuppression >= maxVal] = strong

	# 6) Edge Tracking by Hysteresis
	edges = gradientDoubleThreshold
	for i in range(1, rows - 1):
		for j in range(1, columns - 1):
			if (edges[i, j] == weak):
				try:
					if ((edges[i + 1, j] == strong) or (edges[i, j - 1] == strong) or
							(edges[i, j + 1] == strong) or (edges[i - 1, j] == strong) or
							(edges[i + 1, j + 1] == strong) or (edges[i - 1, j - 1] == strong) or
							(edges[i + 1, j - 1] == strong) or (edges[i - 1, j + 1] == strong)):
						edges[i, j] = strong
					else:
						edges[i, j] = 0
				except IndexError as e:
					pass

	return edges

main()