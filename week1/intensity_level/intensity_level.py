import numpy as np 
import cv2


def check_intensity_level(level):
	""" 
	Intensity level must in [2, 128], and must be power of number two.
	This function checks that condition.

	Args:
		level (int): Intensity level.

	Returns:
		validity (bool): Intensity level is correct or not.

	Examples:
		>>> check_intensity_level(64)
		True
		>>> check_intensity_level(67)
		False

	"""
	return level in map(lambda x: 2 ** x, range(1, 8))


def intensity_level(image, level):
	"""
	Change intensity level for given image according to level argument. 

	Args:
		image (str, numpy.ndarray): Image can be represented as numpy 
					n-dimensional array or string that represents full path
					to the image.
		level (int): Intensity level.

	Returns:
		image (numpy.ndarray): The function returns numpy ndarray
							   if input image is numpy ndarray.
		flag (bool): The function returns correctnes flag if input image
					 is represented as string path to it.

	Raises:
		ValueError: If intensity level doesn't have correct value.
		TypeError: If input image doesn't have correct type.

	Examples:
		>>> intensity_level(np.array([[234, 52], [12, 147]]), 64)
		[[56, 13], [3, 36]]

	"""

	if not check_intensity_level(level):
		raise ValueError("Invalid intensity level. It must be integer between 2 and 128. ")

	# Read image
	if type(image) == str:
		img = cv2.imread(image)
	elif type(image) == np.ndarray:
		img = image
	else:
		raise TypeError("Type of image must string that represents path to the image."
						"Or numpy n-dimensional array. ")

	# Change Intensity Level
	img /= (256 / level)
	img = img.astype(np.int8)

	# Write image
	if type(image) == str:
		name, extension = image.split('.')
		cv2.imwrite(name + "_out." + extension, img)
		return True	
	
	return img


def main():
	pass


if __name__ == "__main__":
	main()