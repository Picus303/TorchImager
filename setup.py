import os
import sys
import shutil
import platform
from skbuild import setup
from setuptools import find_packages
from setuptools.command.install import install

class CleanInstall(install):
    def run(self):
        install.run(self)
        
        install_path = self.install_lib
        extensions = [".cpp", ".h", ".cu", ".hip"]

        for root, dirs, files in os.walk(install_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    os.remove(os.path.join(root, file))

hipcc_path = shutil.which("hipcc")
if hipcc_path is None:
    raise RuntimeError("hipcc not found. Please ensure HIP/ROCm is installed and accessible in your PATH.")

if platform.system() != "Linux":
	raise RuntimeError("This package is only supported on Linux.")

setup(
    name="TorchImager",
    version="0.1.0",
    description="A Python library for displaying 2D tensors using OpenGL and HIP interop",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Picus303",
    author_email="pilleurc@protonmail.com",
    url="https://github.com/Picus303/TorchImager",
    license="MIT",
    packages=find_packages(),
    cmake_install_dir="TorchImager",
    cmake_minimum_required_version="3.5",
    cmake_args=[
        f"-DPYTHON_EXECUTABLE={sys.executable}"
    ],
    include_package_data=True,
    python_requires=">=3.12",
    install_requires=["torch"],
    cmdclass={"install": CleanInstall}
)