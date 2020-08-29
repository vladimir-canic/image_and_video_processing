import unittest
import numpy as np 

from image_rotation import degree2rad
from image_rotation import rotate2d
from image_rotation import pixel_position
from image_rotation import rotate_image
from image_rotation import rotate_image_cv



class TestDegree2Rad(unittest.TestCase):
	

	def setUp(self):

		""" Create testing variable for the function degree to radians. """
		
		pass


	def tearDown(self):
		pass


class TestRotate2D(unittest.TestCase):
	

	def setUp(self):

		""" 
			Create testing variable for the function rotate2d, 
			that actually computes rotation indices. 
		"""
		
		pass


	def tearDown(self):
		pass


class TestPixelPosition(unittest.TestCase):
	

	def setUp(self):

		""" Create testing variable for the function pixel position. """
		pass


	def tearDown(self):
		pass


class TestRotateImage(unittest.TestCase):
	

	def setUp(self):

		""" Create testing variable for the function rotate image. """
		pass


	def tearDown(self):
		pass


class TestRotateImageCV(unittest.TestCase):
	

	def setUp(self):

		""" Create testing variable for OpenCV function rotate image. """
		pass


	def tearDown(self):
		pass


if __name__ == "__main__":
	unittest.main()