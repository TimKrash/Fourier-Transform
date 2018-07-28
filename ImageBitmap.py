from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np 
import os
from copy import deepcopy

class ImageBitmap:

	def __init__(self, img):
		self._img = Image.open(os.getcwd() + '/Images/' + img + '.jpg')

	def image_array(self):
		self._arr = np.array(self._img)
		return self._arr

	# Split the array into three channels (r, g, b)
	def im_bw(self, scale = 3):
		self.r, self.g, self.b = np.split(self._arr, 3, axis = 2)
		self.r = self.r.reshape(-1)
		self.g = self.r.reshape(-1)
		self.b = self.r.reshape(-1)

	# Standard RGB to grayscale conversion
		self.bitmap = list(map(lambda x: 0.299*x[0]+0.587*x[1]+0.114*x[2], zip(self.r, self.g, self.b)))
		self.bitmap = np.array(self.bitmap).reshape(self._arr.shape[0], self._arr.shape[1])
		self.bitmap = np.dot((self.bitmap > 128).astype(float), 255)
		im = Image.fromarray(self.bitmap.astype(np.uint8))
		
	#scale down image
		im_sm = im.resize(tuple([int(v / scale) for v in im.size]), Image.ANTIALIAS)
		im_bw = im_sm.convert(mode='1', dither= 2)
		self.bw_image = im_bw
		self.pixels = (1 - np.asarray(im_bw).astype(int))
		self.pixels_flat = np.reshape(self.pixels, self.pixels.size)

	def bw_on_graph(self):
		print("Dimensions: {}".format(self.bw_image.size))
		print("Number of pixels: {}".format(self.pixels.sum()))
		#plt.imshow(np.asarray(self.bw_image))
		#plt.show()

	def nna(self, start = "random"):

		# get position of each nonzero coord
		nonzero_pos = np.where(self.pixels_flat > 0)[0]

		# create array of the number of nonzero coords
		coord_index = np.array(range(1, len(nonzero_pos) + 1))

		# assign each coordinate a number based on its position
		pixel_index = deepcopy(self.pixels_flat)
		for pos, pix in enumerate(nonzero_pos):
			pixel_index[pix] = pos + 1 # +1 to have pixel correspond to its position (i.e. first to 1, second to 2, etc.)

		# populate the coordinate array for each nonzero coordinate
		im_index = np.reshape(pixel_index, self.pixels.shape)
		self.coords = []
		for x in coord_index:
			x_coord = list([int(c) for c in np.where(im_index == x)])
			self.coords.append(list(x_coord))

		# get the distance between each coordinate
		coord_dis = distance.cdist(self.coords, self.coords, 'euclidean')

		""" nearest neighbor algorithm -- traveling salesman approach """
		cities = self.coords
		city_num = len(cities)

		# determine starting point
		if start == "random":
			start_point = int(np.random.choice(range(city_num), size = 1))
		else:
			assert start < city_num
			start_point = start

		nna_tour = [start_point]
		current_city = start_point

		# the tour
		for i in range(0, city_num):
			dist = deepcopy(coord_dis[current_city, :])
			for end in nna_tour:
				dist[end] = np.inf # convert to floating point number
			nearest_neighbor = np.argmin(dist)
			if nearest_neighbor not in nna_tour:
				nna_tour.append(nearest_neighbor)
			current_city = nearest_neighbor

		y_path = -np.array([cities[nna_tour[d % city_num]] for d in range(city_num + 1) ])[:, 0]
		x_path = np.array([cities[nna_tour[d % city_num]] for d in range(city_num + 1) ])[:, 1]
		y_path = y_path - y_path[0] # extracting mininimum y path
		x_path = x_path - x_path[0] # extracting minimum x path

		# reset tour

		city_num += 1
		np.append(x_path, x_path[0])
		np.append(y_path, y_path[0])

		self.x_path = x_path
		self.y_path = y_path
		self.pixel_num = city_num

		plt.plot(self.x_path, self.y_path)
		plt.show()

mouse = ImageBitmap('mouse')
mouse.image_array()
mouse.im_bw()
mouse.bw_on_graph()
mouse.nna()





