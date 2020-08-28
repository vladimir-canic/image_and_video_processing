import unittest

from intensity_level import *


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
						  self.incorrect_type_message, 
						  self.incorrect_type1)
		self.assertRaises(TypeError, 
						  self.incorrect_type_message, 
						  self.incorrect_type2)
		self.assertRaises(TypeError, 
						  self.incorrect_type_message, 
						  self.incorrect_type3)
		self.assertRaises(TypeError, 
						  self.incorrect_type_message, 
						  self.incorrect_type4)
		self.assertRaises(TypeError, 
						  self.incorrect_type_message, 
						  self.incorrect_type5)
		self.assertRaises(TypeError, 
						  self.incorrect_type_message, 
						  self.incorrect_type6)
		self.assertRaises(TypeError, 
						  self.incorrect_type_message, 
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


		# Variables that are used for checking reading image
		# String, correct

		# String, incorrect

		# Numpy ndarray

		# Other incorrect types


		# Variables for testing correctness of intensity level function
		# Input

		# Output

		pass


	def test_incorrect_level(self):
		pass


	def test_string_correct(self):
		pass


	def test_string_incorrect(self):
		pass


	def test_numpyndarray_correct(self):
		pass


	def test_type_incorrect(self):
		pass


	def test_algorithm_validity(self):
		pass


	def tearDown(self):
		pass



if __name__ == "__main__":
	unittest.main()