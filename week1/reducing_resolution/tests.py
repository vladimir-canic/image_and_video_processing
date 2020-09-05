import unittest

import numpy as np 

from reducing_resolution import reduce_resolution


class TestReduceResolution(unittest.TestCase):


	def setUp(self):
		
		# Variables for testing the correctness of the algorithm

		# Input
		self.input_image1 = np.array([
				[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
				[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
				[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
				[5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
				[5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
				[5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
				[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
				[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
				[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
				[5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
				[5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
				[5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
			])

		self.input_block1 = 3
		self.input_block2 = 4
		self.input_block3 = 6

		# Output
		self.output_image1 = np.array([
					[1, 2, 3, 4],
					[5, 6, 7, 8],
					[1, 2, 3, 4],
					[5, 6, 7, 8]
			])
		self.output_image2 = np.array([
						[2, 3, 4],
						[3, 4, 5],
						[4, 5, 6]
			])
		self.output_image3 = np.array([[3, 5], [3, 5]])


	def test_algorithm(self):
		
		"""Check correctness of the algorithm. """

		self.assertEqual(np.sum(self.output_image1 - 
								reduce_resolution(
									self.input_image1, 
									self.input_block1)), 0)

		self.assertEqual(np.sum(self.output_image2 - 
								reduce_resolution(
									self.input_image1, 
									self.input_block2)), 0)

		self.assertEqual(np.sum(self.output_image3 - 
								reduce_resolution(
									self.input_image1, 
									self.input_block3)), 0)


	def tearDown(self):
		pass



if __name__ == "__main__":
	unittest.main()