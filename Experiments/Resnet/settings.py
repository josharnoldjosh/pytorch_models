"""
Data: different labels must be in folders starting with 0, then 1
"""

config = {
	"data_path":"../data", # don't include a "slash" on the end		

	"batch_size":40,
	"num_epoch":30,
	"learning_rate":0.0001,

	# Resize input images
	"image_width":224,
	"image_height":224
}