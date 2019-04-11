from settings import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

def get_pretrained_model():	
	model_ft = models.resnet18(pretrained=True)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model_ft = model_ft.to(device)

	criterion = nn.BCELoss()

	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

	return model_ft, criterion, optimizer_ft