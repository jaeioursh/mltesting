
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, 3, 1)
		self.conv2 = nn.Conv2d(16, 32, 3, 1)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(4608, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		#x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		#x = self.dropout2(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim=1)
		return output


def train( model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))
			


def main():
	# Training settings
	from torchvision import datasets, transforms
	
	
	device = torch.device("cpu")



	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])
	dataset1 = datasets.MNIST('../data', train=True, download=True,
					   transform=transform)
	train_loader = torch.utils.data.DataLoader(dataset1,32)
	

	model = Net().to(device)
	optimizer = optim.Adadelta(model.parameters(), lr=5e-3)

	
	for epoch in range(100):
		train(model, device, train_loader, optimizer, epoch)
	
		torch.save(model.state_dict(), "mnist_cnn.pt")
	

if __name__ == '__main__':
	main()
