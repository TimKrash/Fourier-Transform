from PIL import Image
import numpy as np 
import os

class ImageBitmap:

	def __init__(self, img):
		self._img = Image.open(os.getcwd() + '/Fourier/Images/' + img + '.jpg')

	def image_array(self):
		self._arr = np.array(self._img)
		return self._arr

	# Split the array into three channels (r, g, b)
	def get_bitmap(self):
		self.r, self.g, self.b = np.split(self._arr, 3, axis = 2)
		self.r = self.r.reshape(-1)
		self.g = self.r.reshape(-1)
		self.b = self.r.reshape(-1)

	# Standard RGB to grayscale conversion
		self.bitmap = list(map(lambda x: 0.299*x[0]+0.587*x[1]+0.114*x[2], zip(self.r, self.g, self.b)))
		self.bitmap = np.array(self.bitmap).reshape(self._arr.shape[0], self._arr.shape[1])
		self.bitmap = np.dot((self.bitmap > 128).astype(float), 255)
		im = Image.fromarray(self.bitmap.astype(np.uint8))
		im.save('image.bmp')






