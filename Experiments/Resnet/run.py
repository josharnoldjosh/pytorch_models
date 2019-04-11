from data import DataLoader
from data import Epoch
from model import get_pretrained_model
from visual import Plot

data = DataLoader()
model, loss, optimizer = get_pretrained_model()
plot = Plot("Baseline")

for epoch in Epoch():

	print("Epoch", epoch)
	
	# train model
	for X, y in data.train:	
		optimizer.zero_grad()
		y_hat = model(X)
		cost = loss(y_hat, y)
		cost.backward()
		optimizer.step()
		plot.train_loss(cost)
		plot.train_acc(y_hat, y)

	# test model on validation set
	for X, y in data.val:
		y_hat = model(X)
		cost = loss(y_hat, y)
		plot.val_loss(cost)
		plot.val_acc(y_hat, y)
		plot.update_cm(y_hat, y)		

	# update plot
	plot.loss() 
	plot.accuracy()
	plot.cm()

print("Script done.")