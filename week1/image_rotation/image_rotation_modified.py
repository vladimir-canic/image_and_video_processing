import time

import numpy as np 
import cv2
import matplotlib.pyplot as plt



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


def get_rotation_matrix(center, angle, scale):

	"""
	Make rotation matrix using rotation angle, center of rotation
	and scaling factor.

	Args:
		center (tuple, int): Tuple that represents center of rotation.
		angle (int): Rotation angle in degreess
		scale (float): Scaling factor, how much we increase or decrease 
					   original object.
 
	Returns:
		rotation_matrix (numpy.ndarray): 

	"""
	
	angle = degree2rad(angle)

	alpha = scale * np.cos(angle)
	betta = scale * np.sin(angle)

	centered_x = (1 - alpha) * center[0] - betta * center[1]
	centered_y = betta * center[0] + (1 - alpha) * center[1]

	rotation_matrix = np.array([
				[ alpha, betta, centered_x], 
				[-betta, alpha, centered_y]
		])

	return rotation_matrix 


def get_indices(shape):

	""" 
	Compute pixel positions for the given dimensions of an image.

	Args:
		shape (tuple, int): Shape of the image.

	Returns:
		indices (numpy.ndarray): Array of positions for each pixel. 
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

	height, width, channel = shape

	try:
		height = int(height)
	except Exception as e:
		raise TypeError("Height of the image must be integer. ")

	try:
		width = int(width)
	except Exception as e:
		raise TypeError("Width of the image must be integer. ")

	try:
		channel = int(channel)
	except Exception as e:
		raise TypeError("Channel of the image must be integer. ")

	return np.indices([height, width, channel]).T.reshape(-1, 3)


def transform_image(indices, rotation_matrix):

	"""
	Transforms given image using rotation matrix.

	Args:
		indices (numpy.ndarray): Indices of the orginal image.
		rotation_matrix (numpy.ndarray): Rotation matrix:

	Returns:
		rotated_indices (numpy.ndarray): Indices of the rotated image.
	"""

	ones = np.ones([1, indices.shape[0]])	
	xy_indices = np.concatenate([indices[:, :2].T, ones])
	rotated_xy_indices = rotation_matrix.dot(xy_indices).T
	channel = indices[:, -1].reshape(indices.shape[0], 1)
	rotated_indices = np.concatenate([rotated_xy_indices, channel], 1).astype(np.int32)

	return rotated_indices


def rotate_image(image, 
				 angle,
				 output_shape, 
				 center, 
				 scale=1.0, 
				 output_path=None):
	"""
	Rotate given image for the given angle.

	Args:
		image(str): String that represents path to the image.
		angle (int): Angle given in degrees, it is intger in range (-inf, inf).
		output_shape (tuple, int): Shape of the output image.
		center (tuple, int): Center of the rotation. If it's not specified
							 the center is central point/pixel of the image.
		scale (flaot): Image scale coefficient.
		output_path (str): Destination for preserving rotated image.

	Returns:
		flag (bool): Flag indicates preserving of the image.

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
		img = img.reshape(img.shape[0], img.shape[1], 1)


	################################################################################
	# Create output image                                                          #
	################################################################################

	rotation_matrix = get_rotation_matrix(center, angle, scale)

	indices = get_indices(img.shape)

	rotated_indices = transform_image(indices, rotation_matrix)

	img_out = np.zeros([output_shape[0], output_shape[1], img.shape[2]])

	#-------------------------------------------------------------------#
	# Remove all indices over boundaries of the output image            #
	#-------------------------------------------------------------------#

	height = output_shape[0] 
	width = output_shape[1]

	height_condition = (rotated_indices[:, 0] >= 0) & \
					   (rotated_indices[:, 0] < height)
	width_condition =  (rotated_indices[:, 1] >= 0) & \
					   (rotated_indices[:, 1] < width)

	condition = height_condition & width_condition

	rotated_indices = rotated_indices[condition]
	indices = indices[condition]

	for i in range(len(rotated_indices)):
		in0, in1, in2 = indices[i]
		out0, out1, out2 = rotated_indices[i]
		img_out[out0, out1, out2] = img[in0, in1, in2]


	################################################################################
	# Write and return the output                                                  #
	################################################################################

	if type(image) == str:
		if output_path is None:
			path_segments = image.split('/')
			name, extension = path_segments[-1].split('.')
			output_path = '/'.join(path_segments[:-1]) + '/' +\
						  name + "_OpenCV_out." + extension
		cv2.imwrite(output_path, img_out)
		return True


def main():

	#########################################################################
	# Test FLAGS                                                            #
	#########################################################################

	TEST_ANGLE_2_DEGREE      = 0
	TEST_GET_ROTATION_MATRIX = 1
	TEST_GET_INDICES         = 0
	TEST_TRANSFORM_IMAGE     = 0
	TEST_WHOLE_ALGORITHM     = 0


	#########################################################################
	# Test Modules Separately                                               #
	#########################################################################

	def image_draw(original_image, rotated_image):
		pass

	#-------------------------------------------------------------#
	# Test Angle Conversion from Degrees into the Radians         #
	#-------------------------------------------------------------#

	if TEST_ANGLE_2_DEGREE:

		for i in range(-6, 7):
			print("Angle Degree: ", 30 * i, 
				  "--> Angle Radians:", degree2rad(30 * i))
		print("\n")

	#-------------------------------------------------------------#
	# Test Get Rotation Matrix                                    #
	#-------------------------------------------------------------#

	def rotataion_matrix_draw(points, center, angle, scale, displ, displ_scale):

		"""
		Auxiliary function for rotating and drawing object.

		Args:
			points (np.ndarray): Represent two dimensional object.
			center (float): Center of the rotation.
			angle (float): Angle of the rotation.
			scale (float): Scale factor of the image.
			displ (float): Displacement parameter for drawing the coordinate
						   system axis through the center of rotation.
			displ_scale (float): Parameter for scaling the axes of the new
								 coordinate system.

		Returns:
			-

		"""

		points = np.concatenate([points, np.ones([1, points.shape[1]])])

		horizontal = np.array([
			[center[0] - displ * displ_scale, center[0] + displ * displ_scale],
			[center[1], center[1]],
			[1, 1]
		])
		vertical = np.array([
			[center[0], center[0]],
			[center[1] - displ * displ_scale, center[1] + displ * displ_scale],
			[1, 1]
		])

		rotation_matrix = get_rotation_matrix(center, angle, scale)

		rotated_points = rotation_matrix.dot(points)
		rotated_horizontal = rotation_matrix.dot(horizontal)
		rotated_vertical = rotation_matrix.dot(vertical)

		# Figure
		plt.figure(figsize=[10, 10])

		# Original Image
		plt.plot(points[0], points[1], 'b', linewidth=5)

		# Rotated Image
		plt.plot(rotated_points[0], rotated_points[1], 'g', linewidth=5)

		# Original Image - Additional Drawings
		plt.plot(points[0], points[1], 'b', marker='o', markersize=10)
		plt.plot([center[0]], [center[1]], 'k', marker='o', markersize=10)
		plt.plot(horizontal[0], horizontal[1], 'b--', linewidth=2)
		plt.plot(vertical[0], vertical[1], 'b--', linewidth=2)

		# Roatated Image - Additional Drawings
		plt.plot(rotated_points[0], rotated_points[1], 'g', marker='o', markersize=10)
		plt.plot([center[0]], [center[1]], 'k', marker='o', markersize=10)
		plt.plot(rotated_horizontal[0], rotated_horizontal[1], 'g--', linewidth=2)
		plt.plot(rotated_vertical[0], rotated_vertical[1], 'g--', linewidth=2)

		# Figure parameters
		plt.xlim(center[0] - displ, center[1] + displ)
		plt.ylim(center[0] - displ, center[1] + displ)
		plt.xlabel("X-axis", fontsize=15)
		plt.ylabel("Y-axis", fontsize=15)
		plt.title("Rotation Example, Custom", fontsize=20)
		plt.legend(["Original", "Rotated"])
		plt.grid(True)
		plt.show()

	if TEST_GET_ROTATION_MATRIX:

		for i in range(-3, 4):
			center = (0, 0)
			angle = 30 * i
			scale = 1.0
			print(get_rotation_matrix(center, angle, scale))
		print("\n")	

		# Example 1
		center = (5, 5)
		scale = 1.0
		displacement = 20
		displ_scale = 0.75
		angle = 30

		points = np.array([
			[0, 10, 10,  0, 0],
			[0,  0, 10, 10, 0]
		])

		rotataion_matrix_draw(points=points, 
							  center=center, 
							  angle=angle, 
							  scale=scale, 
							  displ=displacement, 
							  displ_scale=displ_scale)

		# Example 2
		center = (5, 5 * cv2.sqrt(3)[0, 0] / 3)
		scale = 1.0
		displacement = 20
		displ_scale = 0.75
		angle = 180

		points = np.array([
			[0, 10,                     5, 0],
			[0,  0, cv2.sqrt(3)[0, 0] * 5, 0]
		])

		rotataion_matrix_draw(points=points, 
							  center=center, 
							  angle=angle, 
							  scale=scale, 
							  displ=displacement, 
							  displ_scale=displ_scale)

		# Example 3
		center = (5, 5 * cv2.sqrt(3)[0, 0] / 3)
		scale = 2.0
		displacement = 20
		displ_scale = 0.75
		angle = 180

		points = np.array([
			[0, 10,                     5, 0],
			[0,  0, cv2.sqrt(3)[0, 0] * 5, 0]
		])

		rotataion_matrix_draw(points=points, 
							  center=center, 
							  angle=angle, 
							  scale=scale, 
							  displ=displacement, 
							  displ_scale=displ_scale)
		# Example 4

		# Figure
		plt.figure(figsize=[10, 10])

		colors = ['r', 'g', 'y', 'b']

		for i in range(1, 6):

			center = (5, 5)
			scale = 1.0
			displ = 20
			displ_scale = 0.75
			angle = 60 * i

			points = np.array([
				[0, 20, 20,  0, 0],
				[0,  0, 10, 10, 0]
			])

			points = np.concatenate([points, np.ones([1, points.shape[1]])])

			horizontal = np.array([
				[center[0] - displ * displ_scale, center[0] + displ * displ_scale],
				[center[1], center[1]],
				[1, 1]
			])
			vertical = np.array([
				[center[0], center[0]],
				[center[1] - displ * displ_scale, center[1] + displ * displ_scale],
				[1, 1]
			])

			rotation_matrix = get_rotation_matrix(center, angle, scale)

			rotated_points = rotation_matrix.dot(points)
			rotated_horizontal = rotation_matrix.dot(horizontal)
			rotated_vertical = rotation_matrix.dot(vertical)

			# Original Image
			plt.plot(points[0], points[1], 'b', linewidth=5)

			# Rotated Image
			plt.plot(rotated_points[0], rotated_points[1], colors[0], linewidth=5)

			# Original Image - Additional Drawings
			plt.plot(points[0], points[1], 'b', marker='o', markersize=10)
			plt.plot([center[0]], [center[1]], 'k', marker='o', markersize=10)
			plt.plot(horizontal[0], horizontal[1], 'b--', linewidth=2)
			plt.plot(vertical[0], vertical[1], 'b--', linewidth=2)

			# Roatated Image - Additional Drawings
			plt.plot(rotated_points[0], rotated_points[1], 'g', marker='o', markersize=10)
			plt.plot([center[0]], [center[1]], 'k', marker='o', markersize=10)
			plt.plot(rotated_horizontal[0], rotated_horizontal[1], 'g--', linewidth=2)
			plt.plot(rotated_vertical[0], rotated_vertical[1], 'g--', linewidth=2)

		# Figure parameters
		plt.xlim(center[0] - displ, center[1] + displ)
		plt.ylim(center[0] - displ, center[1] + displ)
		plt.xlabel("X-axis", fontsize=15)
		plt.ylabel("Y-axis", fontsize=15)
		plt.title("Rotation Example, Custom", fontsize=20)
		plt.legend(["Original", "Rotated"])
		plt.grid(True)
		plt.show()

	#-------------------------------------------------------------#
	# Test Get Indices                                            #
	#-------------------------------------------------------------#

	if TEST_GET_INDICES:

		for i in range(5):
			shape = np.random.randint(1, 6, (3, ))
			print("Shape:", shape)
			print(get_indices(shape))
			print("\n")
		print("\n")

	#-------------------------------------------------------------#
	# Test Transform Image                                        #
	#-------------------------------------------------------------#

	if TEST_TRANSFORM_IMAGE:

		for i in range(-3, 4):

			center = (0, 0)
			angle = 30 * i
			scale = 1.0
			rotation_matrix = get_rotation_matrix(center, angle, scale)

			shape = np.random.randint(1, 6, (3, ))
			indices = get_indices(shape)

			print(transform_image(indices, rotation_matrix))
			print("\n")
		print("\n")	


	#########################################################################
	# Test whole Algorithm                                                  # 
	#########################################################################

	if TEST_WHOLE_ALGORITHM:

		img_out = "./data/image005.jpg"
		angle = 30
		output_shape = (2560, 1440)
		center = (640, 360)

		print(rotate_image(image=img_out, 
						   angle=angle,
						   output_shape=output_shape, 
						   center=center, 
						   scale=1.0, 
						   output_path=None))
		print("\n")


if __name__ == "__main__":
	main()