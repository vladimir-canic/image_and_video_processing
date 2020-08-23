import numpy as np 
import cv2


def rotate_image(image, angle):
	"""
	Rotate given image for the given angle.

	Args:

	Returns:

	Raises:

	Examples:

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

	height = image.shape[0]
	width = image.shape[1]
	channel = image.shape[2]

	# Convert rotation angle from degrees to radians
	angle_rad = (angle % 360) * np.pi / 180

	# Recompute dimensions of the output image based on converted angle and 
	# above extracted dimensions od the original image
	height_out = np.abs(height * np.cos(angle_rad)) + np.abs(width * np.sin(angle_rad))
	width_out = np.abs(width * np.cos(angle_rad)) + np.abs(height * np.sin(angle_rad))

	# Create output image
	img_out = np.zeros([height_out, width_out, channel])

	# Get position of each pixel in the image
	# For e.g. if image has dimensions (2, 3, 2), positions are
	# (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), etc.
	idx = np.indices([height, width, channel]).T.reshape(-1, channel)

	# Recompute indices using rotation angle and input image indices
	height_idx_out = idx[:, 0] * np.cos(angle_rad) - idx[:, 1] * np.sin(angle_rad)
	width_idx_out = idx[:, 0] * np.sin(angle_rad) + idx[:, 1] * np.cos(angle_rad)	
	idx_out = np.concatenate([height_idx_out, width_idx_out, channel], 2)

	# Write input image and create rotated image
	img_out[idx_out] = img[idx]

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
	pass


if __name__ == "__main__":
	main()