import torch

from TorchImager import Window

height, width = 512, 512
tensor = torch.rand(height, width)

with Window(width, height, "grayscale", 1) as window:
	window.show(tensor)