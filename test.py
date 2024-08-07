import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

with open("data"+sys.argv[1]+".pkl","rb") as f:
	data=pickle.load(f)

x=data[0]
N=11
for i in range(1,len(data)):
	dx,la,lb,val=data[i]
	plt.subplot(3,N,i)
	plt.title(str(val))
	if i==1:
		plt.ylabel("L1 multiplier")
	plt.imshow(x,vmin=-0.5,vmax=2.5)
	plt.xticks([])
	plt.yticks([])

	plt.subplot(3,N,i+N)
	plt.title(str(round(la,4)))
	if i==1:
		plt.ylabel("Adversarial Error")
	plt.imshow(dx,vmin=-0.5,vmax=2.5)
	plt.xticks([])
	plt.yticks([])

	plt.subplot(3,N,i+2*N)
	plt.title(round(lb/val,4))
	if i==1:
		plt.ylabel("L1 Loss")
	plt.imshow(x+dx,vmin=-0.5,vmax=2.5)
	plt.xticks([])
	plt.yticks([])

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.subplots_adjust(top=0.985,
bottom=0.015,
left=0.034,
right=0.991,
hspace=0.0,
wspace=0.0367)

plt.show()