cmake_minimum_required(VERSION 3.5)
project(Display LANGUAGES HIP CXX)

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(hip REQUIRED)

# Create the library
add_library(Display MODULE TorchImager/main.hip)

# Link the library with the required packages
target_link_libraries(Display OpenGL::GL GLEW::GLEW glfw hip::host hip::device pybind11::module)
target_include_directories(Display PRIVATE ${pybind11_INCLUDE_DIR})

# Set the install destination
install(TARGETS Display LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})