import numpy as np 
import cv2


def reduce_resolution(image, block):
	"""

	Args:
		image (str, numpy.ndarray): 
		block (int): 

	Returns:
		img_out (numpy.ndarray): 
		flag (bool):

	Raises:

	Examples:

	"""	
	
	img = image
	height = img.shape[0]
	width = img.shape[1]
	channel = img.shape[2]
	img = img[:height - (height % block), :width - (width % block)]

	img_out = np.zeros([height // block, width // block, channel])

	for i in range(height // block):
		for j in range(width // block):
			for k in range(channel):
				img_out[i, j, k] = brr[i * block:(i + 1) * block, j * block:(j + 1) * block, k]

	return img_out


def main():
	pass


if __name__ == "__main__":
	main()