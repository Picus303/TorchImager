import torch

from TorchImager import Window

height, width = 512, 512
tensor = torch.rand(3, height, width)

with Window(width, height, "color", 0.5) as window:
	window.show(tensor)