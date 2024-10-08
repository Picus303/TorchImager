import torch

from TorchImager import Window

height, width = 512, 512

try:
	with Window(width, height, "grayscale", 0.5) as window:
		while True:
			tensor = torch.rand(height, width)
			window.update(tensor)

except KeyboardInterrupt:
	print("\nWindow closed by user.")