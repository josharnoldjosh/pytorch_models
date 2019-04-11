from settings import config
from j.osh import *
import random
import torch
from PIL import Image
from torchvision.transforms import ToTensor

class Epoch:
	def __init__(self):
		self.num_epochs = config["num_epoch"]
		self.idx = 0
	
	def __iter__(self):
		return self

	def __next__(self):
		if self.idx < self.num_epochs:
			self.idx += 1
			return self.idx
		raise StopIteration

	def get_cost(self, cost, scale=1000):
		if torch.cuda.is_available():
			return cost.detach().cpu().numpy()*scale
		else:								
			return cost.detach().numpy()*scale			


class Label:
	"""
	A class for holding the images associated with a label.
	"""
	def __init__(self, image_paths, label):
		self.label = label
		self.image_paths = image_paths
		self.split_data()

	def split_data(self, split=0.8):
		"""
		Splits data
		"""
		random.shuffle(self.image_paths) # shuffle array	

		# determine splits	
		length = len(self.image_paths)		
		train_upper = int((length*split)*split)
		val_upper = int(length*split)

		# split data
		self.train_X = self.image_paths[:train_upper]
		self.train_y = len(self.train_X)*[self.label]

		self.val_X = self.image_paths[train_upper:val_upper]
		self.val_y = len(self.val_X)*[self.label]

		self.test_X = self.image_paths[val_upper:]	
		self.test_y = len(self.test_X)*[self.label]	
		return

class BatchLoader:
	"""
	Serves batches out of a data source.
	"""
	def __init__(self, data):
		self.data = list(data)
		random.shuffle(self.data) # shuffle data
		self.batch_size = config["batch_size"]
		self.batch_idx = 0
		return

	def __iter__(self):
		return self

	def __next__(self):
		upper_idx = (self.batch_idx+1)*self.batch_size
		lower_idx = self.batch_idx*self.batch_size

		if lower_idx < len(self.data):
			self.batch_idx += 1
			return self.preprocess_batch(self.data[lower_idx:upper_idx])

		self.batch_idx = 0
		raise StopIteration

	def preprocess_image(self, path):
		image = Image.open(path)		
		image = image.convert("RGB")				
		image = image.resize((config["image_width"], config["image_height"]))
		image = ToTensor()(image)
		image = image.type("torch.FloatTensor")
		return image

	def preprocess_labels(self, labels):
		y = torch.tensor(labels)
		y = y.type("torch.FloatTensor")
		return y

	def preprocess_batch(self, data):
		"""
		Preprocess a batch to go into the model.		
		"""
		image_paths, labels = zip(*data)		

		X = []
		for path in image_paths:
			image = self.preprocess_image(path)
			X.append(image)
		X = torch.stack(X)

		y = self.preprocess_labels(labels)
		
		if torch.cuda.is_available():
			X = X.cuda()
			y = y.cuda()
		return X, y

class DataLoader:	
	"""
	Loads, shuffles and splits data, then returns BatchLoaders that can iterate over the data.
	"""
	def __init__(self):				
		self.load_data()		

	def load_data(self):
		"""
		Load data into the data loader.
		"""
		self.labels, self.train_X, self.val_X, self.test_X = [],[],[],[] # reset labels
		self.train_y, self.val_y, self.test_y = [],[],[] # reset labels

		path = config["data_path"]
		labels = files(path=path, file_type="/")
		for label, label_path in enumerate(labels):			
			image_paths = files(path=label_path, file_type=".jpg")			
			self.labels += [Label(image_paths, label)]

		# Create our datasets
		for label in self.labels:
			self.train_X += label.train_X
			self.train_y += label.train_y
			self.val_X += label.val_X
			self.val_y += label.val_y
			self.test_X += label.test_X
			self.test_y += label.test_y

		self.train = BatchLoader(zip(self.train_X, self.train_y))
		self.val = BatchLoader(zip(self.val_X, self.val_y))
		self.test = BatchLoader(zip(self.test_X, self.test_y))

	def print_data_len(self):
		print(len(self.train), len(self.val), len(self.test))