import unittest
import numpy as np 

from pixel_average import pixel_average_naive


class PixelAveragingNaive(unittest.TestCase):


	def setUp(self):

		""" Create testing variable for the algorithm pixel averaging """

		# Test variables for incorrect image path
		self.incorrect_image_path1 = "Wrong image path."
		self.incorrect_image_path2 = "./data/images"
		self.incorrect_image_path3 = "./data/image001.jpg.jpg"

		# Test variables for incorrect image type
		self.incorrect_image_type1 = 15
		self.incorrect_image_type2 = 45.56
		self.incorrect_image_type3 = []

		# Test variables for incorrect image shape
		self.incorrect_image_shape1 = np.ones([1, 2, 3, 4])
		self.incorrect_image_shape2 = np.ones([10, 10, 3, 4])
		self.incorrect_image_shape3 = np.ones([5, 8, 4, 5, 7])
		self.incorrect_image_shape4 = np.ones([2, 8, 4, 5, 2, 10])
		self.incorrect_image_shape5 = np.ones([20, ])

		# Test varibales for incorrect neighborhood
		self.neighborhood_image = "./data/image001.jpg"
		self.incorrect_neighborhood1 = 1000
		self.incorrect_neighborhood2 = 600
		self.incorrect_neighborhood3 = 541

		# Test variables for incorrect padding types
		self.padding_image = "./data/image001.jpg"
		self.padding_neighborhood = 10
		self.incorrect_padding_type1 = "left-right"
		self.incorrect_padding_type2 = "left-left"
		self.incorrect_padding_type3 = "right-right"
		self.incorrect_padding_type4 = "top-bottom"

		# Test variable for checking correctness of the algorithm
		# Input
		self.image = np.array([
				[171,  14, 125, 235],
		        [193, 186,  34,  22],
		        [ 67, 143,  82, 157],
		        [218, 141, 148, 154]
		])

		self.input_1 = {
			'image': self.image, 
			'neighborhood': 3, 
			'padtype': None,
			'mode': 'constant',
			'padder': None
		}
		self.input_2 = {
			'image': self.image, 
			'neighborhood': 4, 
			'padtype': 'left-top',
			'mode': 'constant',
			'padder': None
		}
		self.input_3 = {
			'image': self.image, 
			'neighborhood': 4, 
			'padtype': 'left-bottom',
			'mode': 'constant',
			'padder': None
		}
		self.input_4 = {
			'image': self.image, 
			'neighborhood': 4, 
			'padtype': 'right-top',
			'mode': 'constant',
			'padder': 100
		}
		self.input_5 = {
			'image': self.image, 
			'neighborhood': 4, 
			'padtype': 'right-bottom',
			'mode': 'constant',
			'padder': 100
		}

		# Output
		self.output1 = np.array([
				[ 62,  80,  68, 46],
		        [ 86, 112, 110, 72],
		        [105, 134, 118, 66],
		        [ 63,  88,  91, 60]
		])
		self.output2 = np.array([
				[63, 89, 62, 40],
				[95, 130, 90, 59],
				[75, 96, 66, 37],
				[49, 69, 51, 33]
		])
		self.output3 = np.array([
				[45,  61, 38, 26],
				[63,  89, 62, 40],
				[95, 130, 90, 59],
				[75,  96, 66, 37]
		])
		self.output4 = np.array([
				[110, 107, 114, 106],
				[120, 120, 130, 115],
				[121, 119, 121, 110],
				[110, 112, 119, 114]
		])
		self.output5 = np.array([
				[110, 107, 111, 101],
				[110, 107, 114, 106],
				[120, 120, 130, 115],
				[121, 119, 121, 110]
		])


	def test_incorrect_image_path(self):

		""" 
			Check flow of the pixel averaging function 
			for incorrect image path. 
		"""
		
		self.assertRaises(IOError, 
						  pixel_average_naive,
						  self.incorrect_image_path1,
						  None)
		
		self.assertRaises(IOError, 
						  pixel_average_naive,
						  self.incorrect_image_path2,
						  None)
		
		self.assertRaises(IOError, 
						  pixel_average_naive,
						  self.incorrect_image_path3,
						  None)


	def test_incorrect_image_type(self):

		"""
			Check flow of the pixel averaging function 
			for different incorrect image types. 
		"""
		
		self.assertRaises(TypeError, 
						  pixel_average_naive, 
						  self.incorrect_image_type1, 
						  None)
		
		self.assertRaises(TypeError, 
						  pixel_average_naive, 
						  self.incorrect_image_type2, 
						  None)
		
		self.assertRaises(TypeError, 
						  pixel_average_naive, 
						  self.incorrect_image_type3, 
						  None)


	def test_image_shape(self):

		"""
			Check flow of the pixel averaging function 
			for different image shapes. 
		"""
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.incorrect_image_shape1,
						  None)
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.incorrect_image_shape2,
						  None)
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.incorrect_image_shape3,
						  None)
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.incorrect_image_shape4,
						  None)
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.incorrect_image_shape5,
						  None)


	def test_neighborhood(self):

		"""
			Check flow of the pixel averaging function 
			for different sizes of neighborhood. 
		"""
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.neighborhood_image,
						  self.incorrect_neighborhood1)
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.neighborhood_image,
						  self.incorrect_neighborhood2)
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.neighborhood_image,
						  self.incorrect_neighborhood3)


	def test_padding_types(self):

		"""
			Check flow of the pixel averaging function 
			for different padding shapes. 
		"""
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.padding_image, 
						  self.padding_neighborhood,
						  self.incorrect_padding_type1)
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.padding_image, 
						  self.padding_neighborhood,
						  self.incorrect_padding_type2)
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.padding_image, 
						  self.padding_neighborhood,
						  self.incorrect_padding_type3)
		
		self.assertRaises(ValueError, 
						  pixel_average_naive, 
						  self.padding_image, 
						  self.padding_neighborhood,
						  self.incorrect_padding_type4)


	def test_algorithm_validity(self):

		"""
			Check flow of the pixel averaging function 
			for different image values. 
		"""
		
		self.assertEqual(np.sum(self.output1 - pixel_average_naive(**self.input_1)), 0)
		self.assertEqual(np.sum(self.output2 - pixel_average_naive(**self.input_2)), 0)
		self.assertEqual(np.sum(self.output3 - pixel_average_naive(**self.input_3)), 0)
		self.assertEqual(np.sum(self.output4 - pixel_average_naive(**self.input_4)), 0)
		# self.assertEqual(np.sum(self.output5 - pixel_average_naive(**self.input_5)), 0)


	def tearDown(self):
		pass


if __name__ == "__main__":
	unittest.main()
