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
	pass


def main():
	pass


if __name__ == "__main__":
	main()