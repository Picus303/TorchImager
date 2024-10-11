from Lenia import LeniaRGB

from TorchImager import Window

# Parameters
width, height = 31, 31
kernels_size = 31
kernels_params = {
	(0, 0): (4, 1),
	(0, 1): (8, 2),
	(0, 2): (12, 3),
	(1, 0): (12, 3),
	(1, 1): (4, 1),
	(1, 2): (8, 2),
	(2, 0): (8, 2),
	(2, 1): (12, 3),
	(2, 2): (4, 1),
}
mu, sigma, alpha = 20.0, 5.0, 0.1

# Create LeniaRGB object
lenia = LeniaRGB(width, height, kernels_size, kernels_params, mu, sigma, alpha)

# Create window
with Window(width, height, "grayscale", 10) as window:
	window.show(lenia.kernels[0, 1])