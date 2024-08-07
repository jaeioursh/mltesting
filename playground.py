from mnist import Net
import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle

print("reading")
transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])
dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
for i in range(10):
	print(dataset[i][1])



def test(x,multiplier):
	dx=torch.normal(0,0.1,size=(1,1,28,28),requires_grad=True)
	model = Net()
	model.load_state_dict(torch.load("mnist_cnn.pt"))
	pred=model.forward(x)
	print(pred)


	opt=optim.Adam([dx],1e-2)

	for i in range(5000):
		#print(i)
		#dx=torch.clamp(dx,-0.5,0.5)
		opt.zero_grad()
		logits=model.forward(x+dx)
		if target==actual:
			lossa=torch.sum(logits*torch.exp(logits))
		else:
			lossa=F.nll_loss(logits, y)
		lossb=torch.mean(torch.abs(dx))*multiplier		
		(lossa+lossb).backward()
		opt.step()
		with torch.no_grad():
			dx[:] = (x+dx).clamp(-.5, 2.5)-x

	X=x.detach().numpy()[0][0]
	dX=dx.detach().numpy()[0][0]

	print(torch.round(torch.exp(model.forward(x)),decimals=3).detach())
	print(torch.round(torch.exp(model.forward(x+dx)),decimals=3).detach())

	return lossa.detach().item(), lossb.detach().item(), dx.detach().numpy()[0][0]


	plt.subplot(1,3,1)
	plt.imshow(X)
	plt.subplot(1,3,2)
	plt.imshow(dX)
	plt.subplot(1,3,3)
	plt.imshow(X+dX)
	plt.show()

for idx,targ in [[3,4],[3,1]]:#[0,5],[0,3],[4,9],[4,4]]:
	x,y=dataset[idx]#4
	actual=y
	target=targ
	print(actual,target)
	y=torch.tensor([4])
	print(y)
	x=x.reshape((1,1,28,28))

	print(y)
	print(x.shape)

	data=[x.detach().numpy()[0][0]]
	for i in [10,5,1,.5,.1,.05,.01,.005,.001,0.005,0.0001]:
		print(i)
		la,lb,dx=test(x,i)
		data.append([dx,la,lb,i])

	with open("data"+str(actual)+str(target)+".pkl","wb") as f:
		pickle.dump(data,f)