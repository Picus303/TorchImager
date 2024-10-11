import time
from Lenia import LeniaRGB

from TorchImager import Window

class FPSCounter:
	def __init__(self, print_interval):
		self.print_interval = print_interval

		self.frames = 0
		self.last_print_time = time.perf_counter()
	
	def update(self):
		self.frames += 1
		if time.perf_counter() - self.last_print_time > self.print_interval:
			print(f"FPS: {self.frames / self.print_interval}")
			self.frames = 0
			self.last_print_time = time.perf_counter()

fps_counter = FPSCounter(3)


# Parameters
width, height = 512, 512
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

# Create window and run simulation
try:
	with Window(width, height, "color", 1) as window:
		while True:
			lenia.step()
			window.update(lenia.data)
			fps_counter.update()

except KeyboardInterrupt:
	pass