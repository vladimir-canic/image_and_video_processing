import numpy as np 
import cv2



def degree2rad():
	pass


def rotate_resize():
	pass


def rotate2d():
	pass


def pixel_position():
	pass


def rotate_image(image, angle):
	"""
	Rotate given image for the given angle.

	Args:
		image(str, numpy.ndarray): Input image given as numpy ndarray or
								   as string that represents path to the image.
		angle (int): Angle given in degrees, it is intger in range (-inf, inf).

	Returns:
		img_out (numpy.ndarray): If image input is numpy ndarray.
		flag (bool): If image input path.

	Raises:
		IOError: If image isn't read successfully.
		TypeError: If image doesn't have appropriate type.

	"""

	################################################################################
	# Read and check input image                                                   #
	################################################################################

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

	################################################################################
	# Create output image with recalculated dimensions for given rotation angle    #                                      #
	################################################################################

	height = img.shape[0]
	width = img.shape[1]
	channel = img.shape[2]

	print(height)
	print(width)
	print(channel)

	# Convert rotation angle from degrees to radians
	angle_rad = (angle % 360) * np.pi / 180

	# Recompute dimensions of the output image based on converted angle and 
	# above extracted dimensions od the original image
	height_out = np.abs(height * np.cos(angle_rad)) + np.abs(width * np.sin(angle_rad))
	height_out = height_out.astype(np.int32)
	width_out = np.abs(width * np.cos(angle_rad)) + np.abs(height * np.sin(angle_rad))
	width_out = width_out.astype(np.int32)

	# Create output image
	img_out = np.zeros([height_out + 1, width_out + 1, channel])

	# Get position of each pixel in the image
	# For e.g. if image has dimensions (2, 3, 2), positions are
	# (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), etc.
	idx = np.indices([height, width, channel]).T.reshape(-1, channel)

	# Recompute indices using rotation angle and input image indices
	height_idx_out = idx[:, 0] * np.cos(angle_rad) - idx[:, 1] * np.sin(angle_rad)
	height_idx_out = height_idx_out.astype(np.int32)
	height_idx_out = height_idx_out.reshape(-1, 1)
	print(np.max(height_out))

	width_idx_out = idx[:, 0] * np.sin(angle_rad) + idx[:, 1] * np.cos(angle_rad)
	width_idx_out = width_idx_out.astype(np.int32)
	width_idx_out = width_idx_out.reshape(-1, 1)
	print(np.max(width_idx_out))

	channel_idx_out = idx[:, 2].reshape(-1, 1)

	idx_out = np.concatenate([height_idx_out, width_idx_out, channel_idx_out], 1)

	print(idx.shape)
	print(idx_out.shape)

	print(img_out.shape)
	print(img.shape)

	print(np.max(idx.reshape(-1)))
	print(np.max(idx_out.reshape(-1)))

	print(idx_out[:10])
	print(idx[:10])

	# Write input image and create rotated image
	# img_out[idx_out] = img[idx]
	try:
		for i in range(idx.shape[0]):
			img_out[idx_out[i]] = img[idx[i]]
	except ValueError:
		print(i)

	################################################################################
	# Write and return the output                                                  #
	################################################################################

	if type(image) == str:
		path_segments = image.split('/')
		name, extension = path_segments[-1].split('.')
		output = '/'.join(path_segments[:-1]) + '/' + name + "_out." + extension
		cv2.imwrite(output, img_out)
		return True

	return img_out


def main():
	
	##################################################
	# Test without OpenCV                            #
	##################################################

	# Example 1
	img_path = "./data/image001.jpg"
	print(rotate_image(img_path, 45))
	print("\n")

	# Example 2

	# Example 3

	# Example 4

	# Example 5

	##################################################
	# Test with OpenCV                               #
	##################################################

	# Example 1

	# Example 2

	# Example 3

	# Example 4

	# Example 5


if __name__ == "__main__":
	main()