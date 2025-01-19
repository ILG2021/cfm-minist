import torch
import torchdiffeq
from torchcfm.models.unet import UNetModel
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10, class_cond=True).to(device)
model.load_state_dict(torch.load("checkpoints/epoch_10.pth")['model_state_dict'])
model.eval()
# 输入数字6
generate_num = torch.tensor([6], device=device)
with torch.no_grad():
	traj = torchdiffeq.odeint(
		lambda t, x: model.forward(t, x, generate_num),
		torch.randn(1, 1, 28, 28, device=device),
		torch.linspace(0, 1, 2, device=device),
		atol=1e-4,
		rtol=1e-4,
		method='dopri5',
	)
	# 提取生成的最终图像
	# 形状 (num_time_steps, batch_size, channels, height, width)
	single_image = traj[-1, 0].view(1, 28, 28).clip(-1, 1)

	# 转换为 PIL 图像
	img = ToPILImage()(single_image)

	# 保存图片
	img.save("single_generated_digit_5.png")

	# 显示图片
	plt.imshow(img, cmap="gray")
	plt.axis("off")  # 隐藏坐标轴
	plt.show()
