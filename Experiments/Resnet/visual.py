import visdom
import subprocess
import torch
from data import Epoch
from sklearn.metrics import confusion_matrix
import numpy as np

class Plot:
	def __init__(self, model_name="Model"):		
		self.vis = visdom.Visdom() # must run python -m visdom.server	

		self.vis.close() # close all windows

		self.model_name = model_name

		self.train_batch_cost = []
		self.val_batch_cost = []
		self.train_batch_acc = []
		self.val_batch_acc = []
		self.val_batch_cm = []

		self.loss_window = self.vis.line(
		Y=torch.zeros((1)).cpu(),
		X=torch.zeros((1)).cpu(),
		opts=dict(xlabel='Epoch',ylabel='Loss',title=model_name+' Loss',legend=['Train Loss', 'Validation Loss']))
		self.loss_idx = 0

		self.acc_window = self.vis.line(
		Y=torch.zeros((1)).cpu(),
		X=torch.zeros((1)).cpu(),
		opts=dict(xlabel='Epoch',ylabel='Accuracy (%)',title=model_name+' Accuracy',legend=['Train Acc', 'Validation Acc']))
		self.acc_idx = 0

		self.conf_mat_window = self.vis.heatmap(
		X=[[0, 0], [0, 0]],
		opts=dict(
		columnnames=['Positive', 'Negative'],
		rownames=['True', 'False'],
		colormap='Electric', title=model_name+' Confusion Matrix'
		))
		return

	def get_conf_mat_data(self, y_hat, y):
		final_y_hat = []
		final_y = []
		if torch.cuda.is_available():
			y_hat = y_hat.detach().cpu().numpy()
			y = y.detach().cpu().numpy()
		else:
			y_hat = y_hat.detach().numpy()
			y = y.detach().numpy()		
		final_y_hat += [1 if x > 0.5 else 0 for x in y_hat]
		final_y += [1 if x > 0.5 else 0 for x in y]
		tn, fp, fn, tp = confusion_matrix(final_y, final_y_hat).ravel()
		# False Positive, False, negative, True positive, true negative 
		return [tp, tn, fp, fn]

	def update_cm(self, y_hat, y):
		self.val_batch_cm.append(self.get_conf_mat_data(y_hat, y))

	def cm(self):		
		i = np.average(np.array(self.val_batch_cm), axis=0)
		data = [[i[0], i[1]], [i[2], i[3]]]
		self.vis.heatmap(X=data, win=self.conf_mat_window, opts=dict(	
		columnnames=['Positive', 'Negative'],
		rownames=['True', 'False'],
		colormap='Electric', title=self.model_name+' Confusion Matrix', update='replace'))
		self.val_batch_cm = []

	def accuracy(self):
		train_acc = sum(self.train_batch_acc)/len(self.train_batch_acc)
		validation_acc = sum(self.val_batch_acc)/len(self.val_batch_acc)

		self.vis.line(X=torch.ones((1,1)).cpu()*self.acc_idx, 
			Y=torch.Tensor([train_acc]).unsqueeze(0).cpu(),
			win=self.acc_window, update='append', name='Train Acc')

		self.vis.line(X=torch.ones((1,1)).cpu()*self.acc_idx, 
			Y=torch.Tensor([validation_acc]).unsqueeze(0).cpu(),
			win=self.acc_window, update='append', name='Validation Acc')		

		self.acc_idx += 1
		self.train_batch_acc = []
		self.val_batch_acc = []

	def calc_acc(self, y_hat, y):
		final_y_hat = []		
		if torch.cuda.is_available():
			y_hat = y_hat.detach().cpu().numpy()
			y = y.detach().cpu().numpy()
		else:
			y_hat = y_hat.detach().numpy()
			y = y.detach().numpy()		
		final_y_hat += [1 if x[0] > 0.5 else 0 for x in y_hat]
		return (sum(1 for a, b in zip(final_y_hat, y) if a == b) / float(len(final_y_hat)))*100

	def train_acc(self, y_hat, y):
		self.train_batch_acc.append(self.calc_acc(y_hat, y))

	def val_acc(self, y_hat, y):
		self.val_batch_acc.append(self.calc_acc(y_hat, y))

	def train_loss(self, cost):
		"""
		Adds internal values to the train loss.
		"""
		self.train_batch_cost += [Epoch().get_cost(cost)]

	def val_loss(self, cost):
		"""
		Adds internal values to the validation loss.
		"""
		self.val_batch_cost += [Epoch().get_cost(cost)]

	def loss(self):
		"""
		Updates the loss graph with internal loss values.
		"""
		train_loss = sum(self.train_batch_cost)/len(self.train_batch_cost)
		validation_loss = sum(self.val_batch_cost)/len(self.val_batch_cost)

		self.vis.line(X=torch.ones((1,1)).cpu()*self.loss_idx, 
			Y=torch.Tensor([train_loss]).unsqueeze(0).cpu(),
			win=self.loss_window, update='append', name='Train Loss')

		self.vis.line(X=torch.ones((1,1)).cpu()*self.loss_idx, 
			Y=torch.Tensor([validation_loss]).unsqueeze(0).cpu(),
			win=self.loss_window, update='append', name='Validation Loss')		

		self.loss_idx += 1
		self.train_batch_cost = []
		self.val_batch_cost = []

	def clear(self):
		return