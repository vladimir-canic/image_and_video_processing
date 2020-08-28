import unittest

import numpy as np

from intensity_level import check_intensity_level
from intensity_level import intensity_level


class CheckIntensityLevel(unittest.TestCase):

	def setUp(self):

		# Test Variables of Incorrect Type
		self.incorrect_type1 = "I'm not an integer, but I should be an integer."
		self.incorrect_type2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		self.incorrect_type3 = None
		self.incorrect_type4 = 10.23
		self.incorrect_type5 = 16.0
		self.incorrect_type6 = "ABCDE ABCDEF ABCD ABC"
		self.incorrect_type7 = [["ABCDEFGH", 1, 2], [1, 2, 3]]

		# Test Variables of Correct Type
		self.correct_type1 = 14
		self.correct_type2 = 8
		self.correct_type3 = 12

		# Test Variables of Correct Value
		self.correct_value1 = 8
		self.correct_value2 = 16
		self.correct_value3 = 32
		self.correct_value4 = 64
		self.correct_value5 = 128

		# Test Variables of Incorrect Value
		self.incorrect_value1 = -8
		self.incorrect_value2 = -16
		self.incorrect_value3 = 15
		self.incorrect_value4 = 141
		self.incorrect_value5 = 129


	def test_correct_type(self):

		""" Check correctness of the function for integer numbers. """

		self.assertFalse(check_intensity_level(self.correct_type1))
		self.assertTrue(check_intensity_level(self.correct_type2))
		self.assertFalse(check_intensity_level(self.correct_type3))


	def test_incorrect_type(self):

		""" Check correctness of the function for non-integer numbers. """

		self.assertRaises(TypeError, 
						  check_intensity_level, 
						  self.incorrect_type1)
		self.assertRaises(TypeError, 
						  check_intensity_level, 
						  self.incorrect_type2)
		self.assertRaises(TypeError, 
						  check_intensity_level, 
						  self.incorrect_type3)
		self.assertRaises(TypeError, 
						  check_intensity_level, 
						  self.incorrect_type4)
		self.assertRaises(TypeError, 
						  check_intensity_level, 
						  self.incorrect_type5)
		self.assertRaises(TypeError, 
						  check_intensity_level, 
						  self.incorrect_type6)
		self.assertRaises(TypeError, 
						  check_intensity_level, 
						  self.incorrect_type7)


	def test_correct_value(self):

		""" Check correctness of the function for correct values. """

		self.assertTrue(check_intensity_level(self.correct_value1))
		self.assertTrue(check_intensity_level(self.correct_value2))
		self.assertTrue(check_intensity_level(self.correct_value3))
		self.assertTrue(check_intensity_level(self.correct_value4))
		self.assertTrue(check_intensity_level(self.correct_value5))


	def test_incorrect_value(self):

		""" Check correctness of the function for incorrect values. """

		self.assertFalse(check_intensity_level(self.incorrect_value1))
		self.assertFalse(check_intensity_level(self.incorrect_value2))
		self.assertFalse(check_intensity_level(self.incorrect_value3))
		self.assertFalse(check_intensity_level(self.incorrect_value4))
		self.assertFalse(check_intensity_level(self.incorrect_value5))


	def tearDown(self):
		pass



class IntensityLevel(unittest.TestCase):

	def setUp(self):

		# Variables that represents incorrect intensity level
		self.incorrect_level1 = "Wrong type!!!"
		self.incorrect_level2 = 15.26
		self.incorrect_level3 = 7
		self.incorrect_level4 = 112

		# Variables for incorrect image path
		self.incorrect_image_path1 = "Wrong Image Path"
		self.incorrect_image_path2 = "./data/image/image_0001.jpg"
		self.incorrect_image_path3 = "I'm, also, wrong image path."

		# Variables for the other incorrect image types
		self.incorrect_image_type1 = 1547
		self.incorrect_image_type2 = -45.788
		self.incorrect_image_type3 = [12, 45, 37]		

		# Variables for testing correctness of intensity level function
		# Input
		self.input_image1 = np.array([[ 99, 206, 239, 189],
								      [230, 118, 144,  73],
								      [  8, 228, 231, 190],
								      [155, 112, 158, 208],
								      [  7, 204, 143, 113]])
		self.input_level1 = 16

		self.input_image2 = np.array([[ 99, 206, 239, 189],
								      [230, 118, 144,  73],
								      [  8, 228, 231, 190],
								      [155, 112, 158, 208],
								      [  7, 204, 143, 113]])
		self.input_level2 = 32

		self.input_image3 = np.array([[[ 99, 206, 239],
								       [189, 230, 118],
								       [144,  73,   8],
								       [228, 231, 190]],

								      [[155, 112, 158],
								       [208,   7, 204],
								       [143, 113, 181],
								       [231,  80,  27]],

								      [[ 44, 205, 203],
								       [ 65, 175,  30],
								       [212,  86, 125],
								       [146, 254, 121]],

								      [[137, 106,  41],
								       [233, 190, 129],
								       [210, 103, 144],
								       [206,   5,  58]],

								      [[  0, 208, 132],
								       [110, 164, 105],
								       [179,  27,  31],
								       [  2,  68,  38]]])
		self.input_level3 = 16

		self.input_image4 = np.array([[[ 99, 206, 239],
								       [189, 230, 118],
								       [144,  73,   8],
								       [228, 231, 190]],

								      [[155, 112, 158],
								       [208,   7, 204],
								       [143, 113, 181],
								       [231,  80,  27]],

								      [[ 44, 205, 203],
								       [ 65, 175,  30],
								       [212,  86, 125],
								       [146, 254, 121]],

								      [[137, 106,  41],
								       [233, 190, 129],
								       [210, 103, 144],
								       [206,   5,  58]],

								      [[  0, 208, 132],
								       [110, 164, 105],
								       [179,  27,  31],
								       [  2,  68,  38]]])
		self.input_level4 = 32

		# Output
		self.output_image1 = np.array([[ 6, 12, 14, 11],
								       [14,  7,  9,  4],
								       [ 0, 14, 14, 11],
								       [ 9,  7,  9, 13],
								       [ 0, 12,  8,  7]])
		self.output_image2 = np.array([[12, 25, 29, 23],
								       [28, 14, 18,  9],
								       [ 1, 28, 28, 23],
								       [19, 14, 19, 26],
								       [ 0, 25, 17, 14]])
		self.output_image3 = np.array([[[ 6, 12, 14],
								        [11, 14,  7],
								        [ 9,  4,  0],
								        [14, 14, 11]],

								       [[ 9,  7,  9],
								        [13,  0, 12],
								        [ 8,  7, 11],
								        [14,  5,  1]],

								       [[ 2, 12, 12],
								        [ 4, 10,  1],
								        [13,  5,  7],
								        [ 9, 15,  7]],

								       [[ 8,  6,  2],
								        [14, 11,  8],
								        [13,  6,  9],
								        [12,  0,  3]],

								       [[ 0, 13,  8],
								        [ 6, 10,  6],
								        [11,  1,  1],
								        [ 0,  4,  2]]])
		self.output_image4 = np.array([[[12, 25, 29],
								        [23, 28, 14],
								        [18,  9,  1],
								        [28, 28, 23]],

								       [[19, 14, 19],
								        [26,  0, 25],
								        [17, 14, 22],
								        [28, 10,  3]],

								       [[ 5, 25, 25],
								        [ 8, 21,  3],
								        [26, 10, 15],
								        [18, 31, 15]],

								       [[17, 13,  5],
								        [29, 23, 16],
								        [26, 12, 18],
								        [25,  0,  7]],

								       [[ 0, 26, 16],
								        [13, 20, 13],
								        [22,  3,  3],
								        [ 0,  8,  4]]])


	def test_incorrect_level(self):

		""" 
			Check flow of the intensity level function 
			for incorrect value for level. 
		"""

		self.assertRaises(TypeError, 
						  intensity_level,
						  None, 
						  self.incorrect_level1)
		self.assertRaises(TypeError, 
						  intensity_level, 
						  None,
						  self.incorrect_level2)


		self.assertRaises(ValueError, 
						  intensity_level, 
						  None,
						  level=self.incorrect_level3)
		self.assertRaises(ValueError, 
						  intensity_level, 
						  None,
						  self.incorrect_level4)


	def test_incorrect_image_path(self):

		""" 
			Check flow of the intensity level function 
			for incorrect image path. 
		"""
		
		self.assertRaises(IOError, 
						  intensity_level,
						  self.incorrect_image_path1,
						  16)
		
		self.assertRaises(IOError, 
						  intensity_level,
						  self.incorrect_image_path2,
						  16)

		self.assertRaises(IOError, 
						  intensity_level,
						  self.incorrect_image_path3,
						  16)


	def test_incorrect_image_type(self):

		"""
			Check flow of the intensity level function 
			for different incorrect image types. 
		"""
		
		self.assertRaises(TypeError, 
						  intensity_level,
						  self.incorrect_image_type1,
						  16)
		
		self.assertRaises(TypeError, 
						  intensity_level,
						  self.incorrect_image_type2,
						  16)
		
		self.assertRaises(TypeError, 
						  intensity_level,
						  self.incorrect_image_type3,
						  16)


	def test_algorithm_validity(self):

		"""
			Check flow of the intensity level function 
			for different image values. 
		"""

		output = intensity_level(self.input_image1, self.input_level1)
		self.assertEqual(np.sum(self.output_image1 - output), 0)

		output = intensity_level(self.input_image2, self.input_level2)
		self.assertEqual(np.sum(self.output_image2 - output), 0)

		output = intensity_level(self.input_image3, self.input_level3)
		self.assertEqual(np.sum(self.output_image3 - output), 0)

		output = intensity_level(self.input_image4, self.input_level4)
		self.assertEqual(np.sum(self.output_image4 - output), 0)
		

	def tearDown(self):
		pass



if __name__ == "__main__":
	unittest.main()