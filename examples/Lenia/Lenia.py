import torch

torch.set_default_device("cuda")


class LeniaRGB:
	def __init__(self, width: int, height: int, kernels_size: int, kernels_params: dict, mu: float, sigma: float, alpha: float):
		self.width = width
		self.height = height
		self.kernels_size = kernels_size
		self.data = torch.rand(3, height, width)
		self.intermediate = torch.empty(3, height, width)
		self.kernels = torch.empty(3, 3, kernels_size, kernels_size)
		self.mu = mu
		self.sigma = sigma
		self.alpha = alpha

		assert kernels_size % 2 == 1, "Kernels size must be odd."

		self.build_kernels(kernels_params)

		pad = (self.kernels_size - 1) // 2
		self.conv2d = torch.nn.Conv2d(3, 3, kernels_size, stride=1, padding=pad, bias=False)
		self.conv2d.weight.data = self.kernels

	def build_kernels(self, params: dict):
		for i in range(3):
			for j in range(3):
				mu, sigma = params[(i, j)]
				x_center, y_center = self.kernels.size(2) // 2, self.kernels.size(3) // 2

				x = torch.arange(self.kernels_size).view(-1, 1).expand(-1, self.kernels_size)
				y = torch.arange(self.kernels_size).view(1, -1).expand(self.kernels_size, -1)
				dist_delta = torch.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) - mu
				self.kernels[i, j] = torch.exp(-dist_delta ** 2 / (2 * sigma ** 2))
	
	def step(self):
		with torch.no_grad():
			self.intermediate = self.conv2d(self.data)
			self.intermediate = (self.intermediate - self.mu) / self.sigma
			self.intermediate = torch.exp(-self.intermediate ** 2)
			self.data = self.alpha * self.intermediate + (1 - self.alpha) * self.data