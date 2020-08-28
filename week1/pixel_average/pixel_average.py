import time

import numpy as np 
import cv2


def pad_with(vector, pad_width, iaxis, kwargs):
	"""
	This function pads a given vector with a given value.

	Args:
		vector(numpy.ndarray): Input vector.
		pad_width (numpy.ndarray): Padding width.
		iaxis (int): Axis for what we pad given vector.
		padder (int): Padding value.

	Returns
		-
	"""
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def pixel_average_naive(image, neighborhood, padtype="left-top", mode='constant', padder=None):

	"""
	This function computer average value for around each pixel in an image.
	The pixels around a pixel are also known as pixel neighborhood. 

	Args:
		image (str, numpy.ndarray): Input image given as string that represents
									image path or as numpy ndarray.
		neighborhood (int): Size of the neighborhood matrix.
		padtype (str): Used for the even neighborhood size. When we pad using the even 
					   neoghborhood, one padded side is for one smaller than other. The 
					   padding type tells what sides are bigger. Valid values are "left-top", 
					   "left-bottom", "right-top", "right-bottom".
		mode (str, func): Look numpy docs: 
						  https://numpy.org/doc/stable/reference/generated/numpy.pad.html
		padder (int): Integer scalar used to pad the image.

	Returns:
		img_out (numpy.ndarray): If input is numpy ndarray.
		flag (bool): If input is image path.

	Raises:
		IOError: 
		TypeError: 

	Examples:
		>>> img = np.array([[232, 67, 155], [23, 74, 198], [247, 85, 112]])
		>>> img
		array([[[232],
		        [ 67],
		        [155]],

		       [[ 23],
		        [ 74],
		        [198]],

		       [[247],
		        [ 85],
		        [112]]])

		>>> pixel_average_naive(img, 3)
		array([[[ 44],
		        [ 83],
		        [ 54]],

		       [[ 80],
		        [132],
		        [ 76]],

		       [[ 47],
		        [ 82],
		        [ 52]]])

	"""
	
	if type(image) == str:
		img = cv2.imread(image)
		if type(img) != np.ndarray:
			raise IOError("Image isn't read successfully.")
	elif type(image) == np.ndarray:
		img = image
	else:
		raise TypeError("Image type must be string that represents path to the image."
						"Or image type must be numpy n-dimensional array.")

	if img.shape > 3 or img.shape < 2:
		ValueError("Dimensions of the input must 2 or 3. ")

	if len(img.shape) == 2:
		img = img.reshape(img.shape[0], img.shape[1], 1)

	# Init output tensor, output image
	img_out = np.zeros(img.shape)

	# Pad input image for neigborhood size, 
	# so we can compute average neighborhood for edge pixels
	bigger = neighborhood // 2
	smaller = (neighborhood - 1) // 2

	# Averaging window size is even
	if neighborhood % 2 == 0:
		if padtype == "left-top":
			left, top = bigger, bigger
			right, bottom = smaller, smaller
		elif padtype == "left-bottom":
			left, bottom = bigger, bigger
			right, top = smaller, smaller
		elif padtype == "right-top":
			right, top = bigger, bigger
			left, bottom = smaller, smaller
		elif padtype == "right-bottom":
			right, bottom = bigger, bigger
			left, top = smaller, smaller
		else:
			raise ValueError("Invalid padding type. ")

	# Averaging window size is odd.
	else:
		left, right, top, bottom = bigger, bigger, bigger, bigger

	padding = [[top, bottom], [left, right], [0, 0]]
	if padder is None:
		pad_img = np.pad(img, padding, mode)
	else:
		channels = []
		for i in range(img.shape[2]):
			channel = np.pad(img[:, :, i], padding[:-1], pad_with, padder=padder)
			channels.append(channel.reshape(channel.shape[0], channel.shape[1], 1))
		pad_img = np.concatenate(channels, 2)

	# Compute neighborhood average and store into the output image
	for channel in range(pad_img.shape[2]):
		for width in range(pad_img.shape[1] - neighborhood + 1):
			for height in range(pad_img.shape[0] - neighborhood + 1):
				img_out[height, width, channel] = np.mean(pad_img[height:height + neighborhood, 
																  width:width + neighborhood, 
																  channel]).astype(np.uint8)
	if type(image) == str:
		path_segments = image.split('/')
		name, extension = path_segments[-1].split('.')
		output = '/'.join(path_segments[:-1]) + '/' + name + "_out." + extension
		cv2.imwrite(output, img_out)
		return True

	return img_out


def pixel_average():
	pass


def main():
	
	####################################################################
	# Test for function Pixel Averaging Naive                          #
	####################################################################

	# Example 1
	img = np.array([[232, 67, 155], [23, 74, 198], [247, 85, 112]])
	img = img.reshape(img.shape[0], img.shape[1], 1)
	print(pixel_average_naive(img, 3))
	print("\n")

	# Example 2
	img = np.array([[232, 67, 155], [23, 74, 198], [247, 85, 112]])
	img = img.reshape(img.shape[0], img.shape[1], 1)
	print(pixel_average_naive(img, 4))
	print("\n")

	# Example 3	
	img = np.array([[232, 67, 155], [23, 74, 198], [247, 85, 112]])
	img = img.reshape(img.shape[0], img.shape[1], 1)
	print(pixel_average_naive(img, 3, padder=20))
	print("\n")

	# Example 4	
	image_path = "./data/image001.jpg"
	start = time.time()
	print(pixel_average_naive(image_path, 21))
	end = time.time()
	print("Execution time:", end - start)
	print("\n")

	# Example 5	
	image_path = "./data/image002.jpg"
	start = time.time()
	print(pixel_average_naive(image_path, 21, padder=20))
	end = time.time()
	print("Execution time:", end - start)
	print("\n")
	
	####################################################################
	# Test for function Pixel Averaging                                #
	####################################################################

	# Example 1

	# Example 2

	# Example 3

	# Example 4

	# Example 5

	# Example 6


if __name__ == "__main__":
	main()