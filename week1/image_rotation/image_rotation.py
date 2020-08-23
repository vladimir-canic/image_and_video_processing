import numpy as np 
import cv2



def degree2rad(angle):
	""" 
	Convert angle from degree into the radians.

	Args:
		angle (int): Angle in degrees.

	Returns:
		anlge (float): Angle in radians.

	Raises:
		TypeError: If input angle is not integer.

	Examples:
		>>> degree2rad(180)
		3.141592653589793
		>>>
		>>>
		>>> degree2rad(540)
		3.141592653589793

	"""
	
	if type(angle) != int:
		raise TypeError("Angle must be given as integer.")

	return (angle % 360) * np.pi / 180


def rotate_resize(height, width, angle):
	""" 
	Resize the image in rotation process. Recompute height and width
	of the image using old dimensions and angle of rotation.

	Args:
		height (int): Heigh of the image.
		width (int): Width of the image.
		angle (float): Rotation angle given in radians, in range [-2pi, 2pi].

	Returns:
		height_out (int): Height of the new output image.
		width_out (int): Width of the new output image.

	Raises:
		ValueError: If rotation angle is not in specified range.

	Examples:
		>>> rotate_resize(100, 100, np.pi / 3)
		(136, 136)
		>>>
		>>>
		>>> rotate_resize(100, 100, np.pi / 4)
		(141, 141)
		>>>
		>>>
		>>> rotate_resize(100, 100, np.pi / 2)
		(100, 100)
		>>>
		>>>
	
	"""

	if not (angle >= -2 * np.pi and angle <= 2 * np.pi):
		raise ValueError("Angle must be in range [-2pi, 2pi]")

	height_out = np.abs(height * np.cos(angle)) + np.abs(width * np.sin(angle))
	height_out = height_out.astype(np.int32)

	width_out = np.abs(width * np.cos(angle)) + np.abs(height * np.sin(angle))
	width_out = width_out.astype(np.int32)

	return height_out, width_out


def rotate2d(rotation_point, height, width, channel, angle):
	""" 
	The function computes indices of rotated image.

	Args:
		rotation_point (tuple, int): The point about which rotation is done.
		height (numpy.ndarray): Array of indices for height dimension.
		width (numpy.ndarray): Array of indices for width dimension.
		channel (numpy.ndarray): Array of indices for channel dimension.
		angle (float): Rotation angle given in radians, in range [-2pi, 2pi].

	Returns:
		idx_out (numpy.ndarray): Recalculated indices for rotated image.

	Raises:
		ValueError: If rotation angle is not in specified range.

	Examples:
		>>> positions = pixel_position(2, 3, 1)
		>>> positions
		array([[0, 0, 0],
		       [1, 0, 0],
		       [0, 1, 0],
		       [1, 1, 0],
		       [0, 2, 0],
		       [1, 2, 0]])
		>>>
		>>>
		>>> height = positions[:, 0]
		>>> height
		array([0, 1, 0, 1, 0, 1])
		>>>
		>>>
		>>> width = positions[:, 1]
		>>> width
		array([0, 0, 1, 1, 2, 2])
		>>>
		>>>
		>>> channel = positions[:, 2]
		>>> channel
		array([0, 0, 0, 0, 0, 0])
		>>>
		>>>
		>>> rotate2d((2 // 2, 3 // 2). height, width, channel, np.pi / 3)
		
	
	"""

	if not (angle >= -2 * np.pi and angle <= 2 * np.pi):
		raise ValueError("Angle must be in range [-2pi, 2pi]")

	x0, y0 = rotation_point

	height_idx_out = x0 + height * np.cos(angle) - width * np.sin(angle)
	height_idx_out = height_idx_out.astype(np.int32)
	height_idx_out = height_idx_out.reshape(-1, 1)

	width_idx_out = y0 + height * np.sin(angle) + width * np.cos(angle)
	width_idx_out = width_idx_out.astype(np.int32)
	width_idx_out = width_idx_out.reshape(-1, 1)

	channel_idx_out = channel.reshape(-1, 1)

	idx_out = np.concatenate([height_idx_out, width_idx_out, channel_idx_out], 1)

	return idx_out


def pixel_position(height, width, channel):
	""" 
	Compute pixel positions for the given dimensions of an image.

	Args:
		height (int): Height of the image.
		width (int): Width of the image.
		channel (int): Number of channels of the image.

	Returns:
		idx (numpy.ndarray): Array of positions for each pixel. 
							 Length of the array is height x width x channel. 

	Raises:
		TypeError: All parameters must be integer.

	Examples:
		>>> pixel_position(2, 2, 2)
		array([[0, 0, 0],
		       [1, 0, 0],
		       [0, 1, 0],
		       [1, 1, 0],
		       [0, 0, 1],
		       [1, 0, 1],
		       [0, 1, 1],
		       [1, 1, 1]])
		>>>
		>>>
		>>> pixel_position(2, 3, 1)
		array([[0, 0, 0],
		       [1, 0, 0],
		       [0, 1, 0],
		       [1, 1, 0],
		       [0, 2, 0],
		       [1, 2, 0]])
	
	"""
	if type(height) != int:
		raise TypeError("Height of the image must be integer. ")
	if type(width) != int:
		raise TypeError("Width of the image must be integer. ")
	if type(channel) != int:
		raise TypeError("Channel of the image must be integer. ")

	return np.indices([height, width, channel]).T.reshape(-1, 3)


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
	angle_rad = degree2rad(angle)

	# Recompute dimensions of the output image based on converted angle and 
	# above extracted dimensions od the original image
	height_out, width_out = rotate_resize(height, width, angle_rad)

	# Create output image
	img_out = np.zeros([height_out + 1, width_out + 1, channel])

	# Get position of each pixel in the image
	# For e.g. if image has dimensions (2, 3, 2), positions are
	# (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), etc.
	idx = pixel_position(height, width, channel)

	# Recompute indices using rotation angle and input image indice
	idx_out = rotate2d((height // 2, width // 2), idx[:, 0], idx[:, 1], idx[:, 2], angle_rad)

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