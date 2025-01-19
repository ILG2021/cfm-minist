import os

import torch
from torchcfm import ConditionalFlowMatcher
from torchcfm.models.unet import UNetModel
from torchvision import datasets
from torchvision.transforms import transforms

os.makedirs("checkpoints", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
n_epochs = 10

trainset = datasets.MNIST("./data", train=True, download=True,
						   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

sigma = 0.0
model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10, class_cond=True).to(device)
optimizer = torch.optim.Adam(model.parameters())
FM = ConditionalFlowMatcher(sigma=sigma)

for epoch in range(n_epochs):
	for i, data in enumerate(train_loader):
		optimizer.zero_grad()
		x1 = data[0].to(device)
		y = data[1].to(device)
		x0 = torch.randn_like(x1)
		t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
		vt = model(t, xt, y)
		loss = torch.mean((vt - ut) ** 2)
		loss.backward()
		optimizer.step()
		print(f"epoch: {epoch}, steps: {i+1}, loss: {loss.item():.4}", end="\r")
	checkpoint_path = os.path.join("checkpoints", f'epoch_{epoch + 1}.pth')
	torch.save({
		'epoch': epoch + 1,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': loss.item()
	}, checkpoint_path)
