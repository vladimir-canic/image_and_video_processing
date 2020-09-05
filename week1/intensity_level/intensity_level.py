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

	Raises:
		TypeError: If type of level is not integer.

	Examples:
		>>> check_intensity_level(64)
		True
		>>> check_intensity_level(67)
		False

	"""

	if type(level) != int:
		raise TypeError("Type of level must be integer. ")

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
		IOError: If opencv library doesn't read image file successfully.

	Examples:
		>>> intensity_level(np.array([[234, 52], [12, 147]]), 64)
		[[56, 13], [3, 36]]

	"""

	if not check_intensity_level(level):
		raise ValueError("Invalid intensity level. It must be integer between 2 and 128. ")

	# Read image
	if type(image) == str:
		img = cv2.imread(image)
		if type(img) != np.ndarray:
			raise IOError("Image isn't read successfully. ")
	elif type(image) == np.ndarray:
		img = image
	else:
		raise TypeError("Type of image must string that represents path to the image."
						"Or numpy n-dimensional array. ")

	# Change Intensity Level
	img //= (256 // level)

	# Write image
	if type(image) == str:
		path_segments = image.split('/')
		name, extension = path_segments[-1].split('.')
		output = '/'.join(path_segments[:-1]) + '/' + name + "_out." + extension
		cv2.imwrite(output, img)
		return True	
	
	return img


def main():
	"""
	This function represents the main function for 
	testing the code for changing intesnity level of an image.

	"""

	###################################################################
	# Example 1                                                       #
	###################################################################
	test_example_image = np.array([
			[234, 214, 217], 
			[123, 147, 186], 
			[65 , 87 , 15 ]
		])
	test_example_level = 8

	print("Example 1")
	print(intensity_level(test_example_image, test_example_level))
	print("\n")


	###################################################################
	# Example 2                                                       #
	###################################################################
	test_example_image = "./data/image001.jpeg"
	test_example_level = 8

	print("Example 2")
	print(intensity_level(test_example_image, test_example_level))
	print("\n")


if __name__ == "__main__":
	main()