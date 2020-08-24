import os
import time

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

	################################################################################
    # Read and check input image                                                   #
    ################################################################################

	if type(image) == str:
		img = cv2.imread(image)
		if type(img) != np.ndarray:
			raise IOError("Image isn't read successfully. ")
	elif type(image) == np.ndarray:
		img = image
	else:
		raise TypeError("Image type must be string that represents "
                    	"path to the image."
                    	"Or image type must be numpy n-dimensional array.")

	if len(img.shape) == 2:
		img = img.reshape(img.shape[0], img.shape[1], 1)


    ################################################################################
    # Reducing Image Resolution                                                    #
    ################################################################################

	height = img.shape[0]
	width = img.shape[1]
	channel = img.shape[2]
	img = img[:height - (height % block), :width - (width % block)]

	img_out = np.zeros([height // block, width // block, channel])

	for i in range(height // block):
		for j in range(width // block):
			for k in range(channel):
				average = np.mean(img[i * block:(i + 1) * block, 
									  j * block:(j + 1) * block, 
									  k]).astype(np.uint8)
				img_out[i, j, k] = average


	################################################################################
    # Write and return the output                                                  #
    ################################################################################

	if type(image) == str:
		path_segments = image.split('/')
		name, extension = path_segments[-1].split('.')
		output = '/'.join(path_segments[:-1]) + \
				 '/' + name + '_' + str(block) + 'x' + str(block) + \
				 '_out.' + extension
		cv2.imwrite(output, img_out)
		return True

	return img_out


def main():
	
	######################################################################
	# Test with numpy ndarray                                            #
	######################################################################

	# Example 1
	img = np.array([[205,  21, 238, 200,  72, 121, 220, 239, 220, 171],
			        [ 30, 139, 132,  65, 176,  73,  44, 167, 231,  30],
			        [205, 226,  17,  45, 131, 205,  73,  45,  29, 204],
			        [ 97, 126, 245,  50, 132,  86,  38, 140, 216, 125],
			        [181, 108, 145, 103, 100, 235, 104, 147,  36, 196],
			        [130, 137, 146, 232, 116,  22,  93, 124, 194, 196],
			        [ 19, 132, 209, 203,   7,  84, 100,  32, 224,  65],
			        [165, 133, 107,  90, 241, 145,  67, 167, 237, 211],
			        [253,   7, 140, 106,   4,  34,  30,   0, 235, 242],
			        [155,  12, 191, 152,  50, 254, 122, 111,  38,   1]])
	print(reduce_resolution(img, 3))
	print("\n")

	# Example 2
	img = np.array([[ 16, 185, 207, 252,  91,  20,  54, 196],
			        [146,  18, 241, 144, 126,  53,  27,  85],
			        [ 27, 250,  88, 134,  49, 203, 148, 144],
			        [ 37, 219, 165, 112,  42, 125,   6,  85],
			        [207, 122, 225, 137,  31, 152, 104, 206],
			        [ 16,  27,  41, 231,  26,  41,  55, 221],
			        [141,   7,  65,   8, 253, 119, 143,  52]])
	print(reduce_resolution(img, 3))
	print("\n")

	# Example 3
	img = np.array([[160, 159, 101,  57, 183, 254],
			        [148,  98, 101,  33, 236, 143],
			        [102, 177, 212,  54, 113,  39],
			        [195, 227, 249, 201,  46,   1],
			        [101,  99, 136,  67,  49, 177],
			        [ 49, 134,  40, 202, 192, 185],
			        [ 97, 152, 241, 157, 192, 161],
			        [ 42, 151,   0, 237, 201,  86],
			        [239, 218,  74,  98, 167, 123]])
	print(reduce_resolution(img, 3))
	print("\n")


	######################################################################
	# Test with images                                                   #
	######################################################################

	#-----------------------------------------------#
	# Block size 3 x 3                              #
	#-----------------------------------------------#

	print (20 * "=", 
		   "Reducing Image Resolution using 3 by 3 block. ", 
		   20 * "=")

	block_size = 3

	for image in sorted(os.listdir("./data/")):
		img_path = "./data/" + image
		print("Reducing image resolution ...")
		start = time.time()
		reduced = reduce_resolution(img_path, block_size)
		end = time.time()
		print("Execution time:", end - start)
		if reduced:
			print("Image", image, "successfully reduced ...")
		print("\n")

	
	


if __name__ == "__main__":
	main()