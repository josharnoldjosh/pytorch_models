from settings import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

def Loss():
	loss = torch.nn.BCELoss()
	if torch.cuda.is_available():
		loss.cuda()
	return loss

def Optimizer(model):	
	return optim.Adam(model.parameters(), lr=config["learning_rate"])	

class Model(nn.Module):
	"""
	The model class
	"""

	def __init__(self):
		"""
		Init the class.
		"""
		super(Model, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.fc = nn.Linear(100352, 1)
		self.norm = nn.Sigmoid()
		
		if torch.cuda.is_available():
			self.layer1.cuda()		
			self.layer2.cuda()	
			self.fc.cuda()			

	def forward(self, x):		
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		out = out.squeeze(1)
		out = self.norm(out)	
		return out		