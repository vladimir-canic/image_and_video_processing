import unittest
import numpy as np 

from image_rotation import degree2rad
from image_rotation import rotate2d
from image_rotation import pixel_position
from image_rotation import rotate_image
from image_rotation import rotate_image_cv2



class TestDegree2Rad(unittest.TestCase):
	

	def setUp(self):

		""" Create testing variable for the function degree to radians. """
		
		# Variables for checking type
		self.degree_type1 = 2.5
		self.degree_type2 = '36'

		# Variables for checking correctness of the function
		self.degree_input1 = 535
		self.degree_input2 = -65
		self.degree_input3 = -500
		self.degree_input4 = 360
		self.degree_input5 = 270

		self.degree_output1 = 175 / 180 * np.pi
		self.degree_output2 = 295 / 180 * np.pi
		self.degree_output3 = 220 / 180 * np.pi
		self.degree_output4 = 0
		self.degree_output5 = 270 / 180 * np.pi


	def test_degree_type(self):

		"""	Check type of the input for the function. """
		
		self.assertRaises(TypeError, degree2rad, self.degree_type1)
		self.assertRaises(TypeError, degree2rad, self.degree_type2)


	def test_degree_value(self):
		
		""" Check correctness of the function. """

		self.assertTrue(np.abs(self.degree_output1 - degree2rad(self.degree_input1)) < 1e-6)
		self.assertTrue(np.abs(self.degree_output2 - degree2rad(self.degree_input2)) < 1e-6)
		self.assertTrue(np.abs(self.degree_output3 - degree2rad(self.degree_input3)) < 1e-6)
		self.assertTrue(np.abs(self.degree_output4 - degree2rad(self.degree_input4)) < 1e-6)
		self.assertTrue(np.abs(self.degree_output5 - degree2rad(self.degree_input5)) < 1e-6)


	def tearDown(self):
		pass


class TestRotate2D(unittest.TestCase):
	

	def setUp(self):

		""" 
			Create testing variable for the function rotate2d, 
			that actually computes rotation indices. 
		"""
		
		# Variables for testing the rotation angle values
		self.rotation_angle1 = ...
		self.rotation_angle2 = ...
		self.rotation_angle3 = ...
		self.rotation_angle4 = ...
		self.rotation_angle5 = ...

		# Variables for testing correctness of the function
		# Input
		self.shape_input1 = ...
		self.angle_input1 = ...

		self.shape_input2 = ...
		self.angle_input2 = ...
		
		self.shape_input3 = ...
		self.angle_input3 = ...

		# Output 
		self.output1 = ...
		self.output2 = ...
		self.output3 = ...


	def test_angle(self):

		""" Check value of the angle. """
		pass


	def test_algorithm(self):

		""" Check correctness of the algorithm. """
		pass


	def tearDown(self):
		pass


class TestPixelPosition(unittest.TestCase):
	

	def setUp(self):

		""" Create testing variable for the function pixel position. """
		
		# Variables for checking dimension type
		self.height_type_incorrect = -12.45
		self.height_type_correct = 6

		self.width_type_incorrect = -12.45
		self.width_type_correct = 5

		self.channel_type_incorrect = -12.45
		self.channel_type_correct = 8

		# Variables for checking correctness of the algorithm
		self.input1 = [2, 2, 2]
		self.input2 = [3, 2, 1]
		self.input3 = [3, 4, 2]

		self.output1 = np.array([
				[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
				[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
			])
		self.output2 = np.array([
				[0, 0, 0], [1, 0, 0], [2, 0, 0],
				[0, 1, 0], [1, 1, 0], [2, 1, 0]
			])
		self.output3 = np.array([
				[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0],
				[1, 1, 0], [2, 1, 0], [0, 2, 0], [1, 2, 0],
				[2, 2, 0], [0, 3, 0], [1, 3, 0], [2, 3, 0],
				[0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1],
				[1, 1, 1], [2, 1, 1], [0, 2, 1], [1, 2, 1],
				[2, 2, 1], [0, 3, 1], [1, 3, 1], [2, 3, 1],
			])


	def test_dimension_type(self):
		
		""" Check type of image dimensions. """

		self.assertRaises(TypeError, 
						  pixel_position, 
						  self.height_type_incorrect,
						  self.width_type_correct,
						  self.channel_type_correct)

		self.assertRaises(TypeError, 
						  pixel_position, 
						  self.height_type_correct,
						  self.width_type_incorrect,
						  self.channel_type_correct)

		self.assertRaises(TypeError, 
						  pixel_position, 
						  self.height_type_correct,
						  self.width_type_correct,
						  self.channel_type_incorrect)

	def test_pixel_position(self):
		
		""" Check correctness of the algorithm. """

		self.assertEqual(np.sum(self.output1 - pixel_position(*self.input1)), 0)
		self.assertEqual(np.sum(self.output2 - pixel_position(*self.input2)), 0)
		self.assertEqual(np.sum(self.output3 - pixel_position(*self.input3)), 0)


	def tearDown(self):
		pass


class TestRotateImage(unittest.TestCase):
	

	def setUp(self):

		""" Create testing variable for the function rotate image. """
		
		# Variables for testing the image type
		self.image_type1 = "Wrong string"
		self.image_type2 = "./data/"
		self.image_type3 = -15.36
		self.image_type4 = [1, 56, 32]

		# Variables for testing the image shape
		self.image_shape1 = (5, 4, 3, 5)
		self.image_shape2 = (5, )
		self.image_shape3 = (6, 5, 3, 7, 6)

		# Variables for testing the correctness of algorithms
		# Input
		self.image_input1 = np.array([
				[],
				[],
				[],
				[],
				[]
			])
		self.angle_input1 = 45

		self.image_input2 = np.array([

			])
		self.angle_input2 = 30

		self.image_input3 = np.array([

			])
		self.angle_input3 = 90

		# Output
		self.output1 = np.array([

			])
		self.output2 = ...
		self.output3 = ...


	def test_image_type(self):
		pass


	def test_image_shape(self):
		pass


	def test_rotation(self):
		pass


	def tearDown(self):
		pass


class TestRotateImageCV(unittest.TestCase):
	

	def setUp(self):

		""" Create testing variable for OpenCV function rotate image. """
		pass


	def tearDown(self):
		pass


if __name__ == '__main__':
    unittest.main()
