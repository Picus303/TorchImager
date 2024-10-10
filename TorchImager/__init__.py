import time
import torch
import TorchImager.libDisplay as ld

# Frame refresh delay (30 FPS)
REFRESH_DELAY = 1 / 30


class Window:
    """
    A class for displaying 2D tensors using OpenGL and HIP interop.
    Supports both grayscale and color displays.

    Attributes:
    -----------
    width : int
        The width of the window in pixels.
    height : int
        The height of the window in pixels.
    type : str
        The type of the window ('grayscale' or 'color').
    scale : float
        The scale factor for enlarging or shrinking the window.

    Methods:
    --------
    initialize():
        Initializes the OpenGL window and the necessary resources.
    update(tensor: torch.Tensor):
        Updates the window content with the provided tensor data.
    show(tensor: torch.Tensor):
        Displays the tensor in the window and keeps the window open until interrupted.
    close():
        Closes the window and releases any associated resources.
    """

    def __init__(self, width: int, height: int, type: str, scale: float = 1.0, auto_norm: bool = False):
        """
        Initializes the Window object with the given parameters.

        Parameters:
        -----------
        width : int
            The width of the window.
        height : int
            The height of the window.
        type : str
            The type of the window ('grayscale' or 'color').
        scale : float, optional
            The scale factor for enlarging or shrinking the window (default is 1.0).
        auto_norm : bool, optional
            Whether to normalize the tensor data to the range [0, 1] before displaying it (default is False).

        Raises:
        -------
        AssertionError
            If any of the provided parameters are invalid.
        """
        self.width = width
        self.height = height
        self.type = type
        self.scale = scale
        self.auto_norm = auto_norm

        # Ensure the input parameters are valid
        assert self.width > 0, "Width must be greater than 0."
        assert self.height > 0, "Height must be greater than 0."
        assert self.scale > 0, "Scale must be greater than 0."
        assert self.type in ["grayscale", "color"], "Type must be 'grayscale' or 'color'."

        self.initialize()

    def __str__(self):
        """Returns a string representation of the Window object."""
        return f"Window(width={self.width}, height={self.height}, type={self.type}, scale={self.scale})"

    def __del__(self):
        """Ensures the graphic ressources are released when the object is deleted."""
        self.close()

    def initialize(self):
        """
        Initializes the OpenGL window and HIP resources.

        Raises:
        -------
        RuntimeError
            If the initialization of the window or resources fails.
        """
        is_color = True if self.type == "color" else False
        self.window = ld.DisplayWindow(self.width, self.height, self.scale, is_color)

        if not self.window.initialize():
            raise RuntimeError("Failed to initialize window. (C++ error)")

    def update(self, tensor: torch.Tensor):
        """
        Updates the window content with the provided tensor.

        Parameters:
        -----------
        tensor : torch.Tensor
            The tensor containing the image data to display. Must be of type torch.float32.

        Raises:
        -------
        AssertionError
            If the tensor does not match the expected shape or data type.
        """
        size = tensor.size()

        # Ensure the tensor has the correct type and dimensions
        assert tensor.dtype == torch.float32, "Tensor must be of type torch.float32."

        if self.type == "grayscale":
            assert len(size) == 2, "Grayscale tensor must have 2 dimensions."
            assert size[0] == self.height, f"Grayscale tensor height must be {self.height}."
            assert size[1] == self.width, f"Grayscale tensor width must be {self.width}."
        elif self.type == "color":
            assert len(size) == 3, "Color tensor must have 3 dimensions."
            assert size[0] == 3, "Color tensor must have 3 channels."
            assert size[1] == self.height, f"Color tensor height must be {self.height}."
            assert size[2] == self.width, f"Color tensor width must be {self.width}."

        # Normalize the tensor data to the range [0, 1]
        if self.auto_norm:
            normalized_tensor = tensor.clone()
            normalized_tensor -= normalized_tensor.min()
            normalized_tensor /= normalized_tensor.max()
        else:
            normalized_tensor = tensor
        
        # Reorder the tensor dimensions for display
        if self.type == "color":
            normalized_tensor = normalized_tensor.permute(1, 2, 0)

        # Ensure the tensor is contiguous in memory
        if not normalized_tensor.is_contiguous():
            normalized_tensor = normalized_tensor.contiguous()

        # Update the window with the tensor's data pointer
        torch.cuda.synchronize()
        self.window.update(normalized_tensor.data_ptr())

    def show(self, tensor: torch.Tensor):
        """
        Displays the tensor in the window and keeps the window open until interrupted.

        Parameters:
        -----------
        tensor : torch.Tensor
            The tensor containing the image data to display.

        Raises:
        -------
        KeyboardInterrupt
            Allows the user to press Ctrl+C to stop showing the window and close it.
        """
        self.update(tensor)

        try:
            print("Press Ctrl+C to continue...")
            while True:
                # Continuously refresh the window
                self.window.refresh()
                time.sleep(REFRESH_DELAY)	# limit resource usage
        except KeyboardInterrupt:
            print("\nWindow closed by user.")

    def close(self):
        """
        Closes the window and releases any associated resources.
        """
        if hasattr(self, "window"):
            self.window.close()
            del self.window

    def __enter__(self):
        """
        Enables the 'with' statement usage, ensuring the window is properly closed after use.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Ensures the window is closed when the 'with' block exits.
        """
        self.close()