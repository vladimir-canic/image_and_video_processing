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

	# Create output image with recalculated 
	# dimensions for given rotation angle
	height = image.shape[0]
	width = image.shape[1]
	channel = image.shape[2]

	angle_rad = angle * np.pi / 180
	height_out = np.abs(height * np.cos(angle_rad)) + np.abs(width * np.sin(angle_rad))
	width_out = np.abs(width * np.cos(angle_rad)) + np.abs(height * np.sin(angle_rad))

	img_out = np.zeros([height_out, width_out, channel])
	idx = np.indices([height, width, channel]).T.reshape(-1, channel)
	height_idx_out = idx[:, 0] * np.cos(angle_rad) - idx[:, 1] * np.sin(angle_rad)
	width_idx_out = idx[:, 0] * np.sin(angle_rad) + idx[:, 1] * np.cos(angle_rad)
	idx_out = np.concatenate([height_idx_out, width_idx_out, channel], 2)

	img_out[idx_out] = image[idx]


def main():
	pass


if __name__ == "__main__":
	main()