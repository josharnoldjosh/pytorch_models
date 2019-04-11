from j.osh import *

data = files(file_type='.jpg')

ensure_dir("output")

for i in range(0, len(data)):
	copy(data[i], "output/"+str(i)+".jpg")

print("script done.")
	