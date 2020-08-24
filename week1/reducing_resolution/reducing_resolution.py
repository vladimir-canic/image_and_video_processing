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
                raise IOError("Image isn't read successfully.")
    elif type(image) == np.ndarray:
        img = image
    else:
        raise TypeError("Image type must be string that represents "
                        "path to the image."
                        "Or image type must be numpy n-dimensional array.")

    if len(img.shape) == 2:
            img = img.reshape(-1, 1)


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
				img_out[i, j, k] = brr[i * block:(i + 1) * block, 
									   j * block:(j + 1) * block, k]


	################################################################################
    # Write and return the output                                                  #
    ################################################################################

    if type(image) == str:
        path_segments = image.split('/')
        name, extension = path_segments[-1].split('.')
        output = '/'.join(path_segments[:-1]) + 
        		 '/' + name + 'OpenCV_out.' + extension
        cv2.imwrite(output, img_out)
        return True

	return img_out


def main():
	
	##############################################################
	# Test with numpy ndarray                                    #
	##############################################################

	# Example 1
	img = ...
	print(...)
	print("\n")

	# Example 2
	img = ...
	print(...)
	print("\n")

	# Example 3
	img = ...
	print(...)
	print("\n")


	##############################################################
	# Test with numpy ndarray                                    #
	##############################################################

	# Example 1
	img_path = ...
	print(...)
	print("\n")

	# Example 2
	img_path = ...
	print(...)
	print("\n")

	# Example 3
	img_path = ...
	print(...)
	print("\n")

	# Example 4
	img_path = ...
	print(...)
	print("\n")

	# Example 5
	img_path = ...
	print(...)
	print("\n")


if __name__ == "__main__":
	main()