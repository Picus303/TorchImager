cmake_minimum_required(VERSION 3.5)
project(Display LANGUAGES CXX)

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

if (USE_CUDA)
	enable_language(CUDA)
	find_package(CUDA REQUIRED)

	add_library(Display MODULE TorchImager/main.cu)

	target_link_libraries(Display OpenGL::GL GLEW::GLEW glfw CUDA::cudart CUDA::cuda_driver pybind11::module)

elseif(USE_HIP)
	enable_language(HIP)
	find_package(hip REQUIRED)

	add_library(Display MODULE TorchImager/main.hip)

	target_link_libraries(Display OpenGL::GL GLEW::GLEW glfw hip::host hip::device pybind11::module)

else()
	message(FATAL_ERROR "Neither CUDA nor HIP were selected. Please specify -DUSE_CUDA=ON or -DUSE_HIP=ON when configuring.")
endif()

# Set the install destination
install(TARGETS Display LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})