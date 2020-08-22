import numpy as np 
import cv2


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def pixel_average_naive(image, neighborhood, padtype="left-top", mode='constant', padder=None):

	"""
	This function computer average value for around each pixel in an image.
	The pixels around a pixel are also known as pixel neighborhood. 

	Args:
		image (str, numpy.ndarray):
		neighborhood (int): 
		padtype (str): 
		mode (str, func): 
		padder (int):

	Returns:
		img_out (numpy.ndarray):
		flag (bool):

	Raises:
		IOError: 
		TypeError: 

	Examples:
		>>> img = np.array([[232, 67, 155], [23, 74, 198], [247, 85, 112]])
		>>> img
		[[], [], []]
		>>> pixel_average_naive()
		[[], [], []]

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

	if len(img.shape) == 2:
		img = img.reshape(-1, 1)

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
	if padder is not None:
		pad_img = np.pad(img, padding, mode)
	else:
		pad_img = np.pad(img, padding, pad_with, padder=padder)

	# Compute neighborhood average and store into the output image
	for channel in range(pad_img.shape[2]):
		for width in range(pad_img.shape[1] - neighborhood + 1):
			for height in range(pad_img.shape[0] - neighborhood + 1):
				img_out[height, width, channel] = np.mean(pad_img[height:height + neighborhood, 
																  width:width + neighborhood, 
																  channel])
	return img_out


def pixel_average():
	pass


def main():
	pass


if __name__ == "__main__":
	main()