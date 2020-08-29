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


def rotate2d(shape, angle):
	""" 
	The function computes indices of rotated image.

	Args:
		shape (iter, int): Shape of the image.
		angle (float): Rotation angle given in radians, in range [-2pi, 2pi].

	Returns:
		idx (numpy.ndarray): Indices of the input image.
		idx_out (numpy.ndarray): Recalculated indices of the rotated image.

	Raises:
		ValueError: If rotation angle is not in specified range.
	
	"""

	if not (angle >= -2 * np.pi and angle <= 2 * np.pi):
		raise ValueError("Angle must be in range [-2pi, 2pi]")

	height, width, channel = shape

	# Get position of each pixel in the image
	# For e.g. if image has dimensions (2, 3, 2), positions are
	# (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), etc.
	idx = pixel_position(height, width, channel)

	# Get rotation point
	x0, y0 = height // 2, width // 2

	# Get indices of the input image
	x1, y1 = idx[:, 0], idx[:, 1]

	################################################################################
	# Compute indices of the output image                                          #
	################################################################################

	# Height coordinate
	height_idx_out = x0 + np.cos(angle) * (x1 - x0) - np.sin(angle) * (y1 - y0)

	height_idx_out = height_idx_out.astype(np.int32)
	height_idx_out = height_idx_out.reshape(-1, 1)

	# Width coordinate
	width_idx_out = y0 + np.sin(angle) * (x1 - x0) + np.cos(angle) * (y1 - y0)

	width_idx_out = width_idx_out.astype(np.int32)
	width_idx_out = width_idx_out.reshape(-1, 1)

	# Channel dimension is unchanged
	channel_idx_out = idx[:, 2].reshape(-1, 1)

	idx_out = np.concatenate([height_idx_out, width_idx_out, channel_idx_out], 1)

	################################################################################
	# Remove all indices over boundaries of the input image                        #
	################################################################################

	height_condition = (idx_out[:, 0] >= 0) & (idx_out[:, 0] < height)
	width_condition =  (idx_out[:, 1] >= 0) & (idx_out[:, 1] < width)

	condition = height_condition & width_condition

	return idx[condition], idx_out[condition]


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
	    raise TypeError("Image type must be string that represents "
	    				"path to the image."
	    				"Or image type must be numpy n-dimensional array.")

	if len(img.shape) > 3 or len(img.shape) < 2:
		raise ValueError("Image must have 2 or 3 dimensions. ")

	if len(img.shape) == 2:
		img = img.reshape(img.shape[0], img.shape[1], 1)

	################################################################################
	# Create output image with recalculated dimensions for given rotation angle    #                                      #
	################################################################################

	# Convert rotation angle from degrees to radians
	angle_rad = degree2rad(angle)

	# Create output image
	img_out = np.zeros(img.shape)

	# Recompute indices using rotation angle and input image indice
	idx, idx_out = rotate2d(img.shape, angle_rad)

	# Write input image and create rotated image
	for i in range(idx.shape[0]):
		try:
			in0, in1, in2 = idx[i]
			out0, out1, out2 = idx_out[i]
			img_out[out0, out1, out2] = img[in0, in1, in2]
		except ValueError:
			print(i)
		except IndexError:
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


def rotate_image_cv2(image, angle):
	"""
	Rotate given image for the given angle.

	Args:
		image(str): String that represents path to the image.
		angle (int): Angle given in degrees, it is intger in range (-inf, inf).

	Returns:
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
	else:
	    raise TypeError("Image type must be string that "
	    				"represents path to the image."
	    				"Or image type must be numpy n-dimensional array.")

	if len(img.shape) == 2:
		img = img.reshape(-1, 1)

	################################################################################
	# Create output image                                                          #
	################################################################################

	image_center = tuple(np.array(img.shape[1::-1]) // 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	img_out = cv2.warpAffine(img, 
							 rot_mat, 
							 img.shape[1::-1], 
							 flags=cv2.INTER_LINEAR)

	################################################################################
	# Write and return the output                                                  #
	################################################################################

	if type(image) == str:
		path_segments = image.split('/')
		name, extension = path_segments[-1].split('.')
		output = '/'.join(path_segments[:-1]) + '/' + name + "OpenCV_out." + extension
		cv2.imwrite(output, img_out)
		return True


def main():
	
	##################################################
	# Test without OpenCV                            #
	##################################################

	# Example 1
	img_path = "./data/image001.jpg"
	print(rotate_image(img_path, 30))
	print("\n")

	# Example 2
	img_path = "./data/image002.jpg"
	print(rotate_image(img_path, 60))
	print("\n")

	# Example 3
	img_path = "./data/image003.jpg"
	print(rotate_image(img_path, 90))
	print("\n")

	# Example 4
	img_path = "./data/image004.jpg"
	print(rotate_image(img_path, -60))
	print("\n")

	# Example 5
	img_path = "./data/image005.jpg"
	print(rotate_image(img_path, -30))
	print("\n")

	##################################################
	# Test with OpenCV                               #
	##################################################

	# Example 1
	img_path = "./data/image001.jpg"
	print(rotate_image_cv2(img_path, 30))
	print("\n")

	# Example 2
	img_path = "./data/image002.jpg"
	print(rotate_image_cv2(img_path, 60))
	print("\n")

	# Example 3
	img_path = "./data/image003.jpg"
	print(rotate_image_cv2(img_path, 90))
	print("\n")

	# Example 4
	img_path = "./data/image004.jpg"
	print(rotate_image_cv2(img_path, -60))
	print("\n")

	# Example 5
	img_path = "./data/image005.jpg"
	print(rotate_image_cv2(img_path, -30))
	print("\n")


if __name__ == "__main__":
	main()