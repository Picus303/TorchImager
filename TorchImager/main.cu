#include <Python.h>
#include <pybind11/pybind11.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <mutex>

/// Macro to handle CUDA errors and print the error message
#define CUDA_ASSERT(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    }

class DisplayWindow {
public:
    // Constructor to initialize window dimensions, scaling factor, and color mode
    DisplayWindow(int width, int height, float scaleFactor, bool isColor)
        : width(width), height(height), scaleFactor(scaleFactor), isColor(isColor),
          window(nullptr), pbo(0), texture(0), cudaResource(nullptr), d_buffer(nullptr) {}

    // Destructor to ensure resources are released when the object is destroyed
    ~DisplayWindow() {
        close();  // Ensure resources are properly released when the window is closed
    }

    // Method to initialize GLFW, GLEW, and create the window
    bool initialize() {
        // Lock the mutex to ensure thread-safe initialization of GLFW
        std::lock_guard<std::mutex> lock(glfwMutex);

        // Initialize GLFW once per application
        if (!glfwInitialized) {
            if (!glfwInit()) {
                std::cerr << "Failed to initialize GLFW" << std::endl;
                return false;
            }
            glfwInitialized = true;
        }

        // Set window size based on the scaling factor
        int windowWidth = width * scaleFactor;
        int windowHeight = height * scaleFactor;

        // Create a GLFW window
        window = glfwCreateWindow(windowWidth, windowHeight, isColor ? "Color Tensor Display" : "Grayscale Tensor Display", NULL, NULL);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            return false;
        }
        glfwMakeContextCurrent(window);  // Set the OpenGL context to the newly created window

        windowCount++;  // Increment window count for proper GLFW management

        // Initialize GLEW to load OpenGL extensions
        if (glewInit() != GLEW_OK) {
            std::cerr << "Failed to initialize GLEW" << std::endl;
            return false;
        }

        // Check compatibility between OpenGL and CUDA
        unsigned int deviceCount;
        int devices[1];  // We only care about one device for now
        CUDA_ASSERT(cudaGLGetDevices(&deviceCount, devices, 1, cudaGLDeviceListAll));
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices compatible with OpenGL found" << std::endl;
            return false;
        }
        CUDA_ASSERT(cudaSetDevice(devices[0]));  // Set the CUDA device

        // Determine the number of channels and OpenGL texture format
        int numChannels = isColor ? 3 : 1;  // 3 channels for RGB, 1 for grayscale
        GLenum format = isColor ? GL_RGB : GL_LUMINANCE;  // Use GL_RGB for color, GL_LUMINANCE for grayscale

        // Create a Pixel Buffer Object (PBO) to share data between OpenGL and CUDA
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * numChannels * sizeof(float), NULL, GL_DYNAMIC_DRAW);

        // Register the PBO with CUDA for interoperability
        CUDA_ASSERT(cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

        // Create an OpenGL texture to display the tensor data
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_FLOAT, NULL);

        // Use nearest-neighbor filtering to avoid color distortion when scaling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Enable 2D texturing in OpenGL
        glEnable(GL_TEXTURE_2D);

        // Map the CUDA resource once during initialization
        CUDA_ASSERT(cudaGraphicsMapResources(1, &cudaResource, 0));

        // Obtain a pointer to the GPU buffer for the mapped resource
        size_t num_bytes;
        CUDA_ASSERT(cudaGraphicsResourceGetMappedPointer((void**)&d_buffer, &num_bytes, cudaResource));

        return true;
    }

    // Method to update the displayed content with new tensor data
    bool update(uintptr_t data_ptr) {
        if (!window) {
            return false;  // Ensure the window is valid
        }

        // Set the current OpenGL context to the window before rendering
        glfwMakeContextCurrent(window);

        // Determine the number of channels
        int numChannels = isColor ? 3 : 1;

        // Copy the GPU tensor data into the PBO (shared with OpenGL)
        CUDA_ASSERT(cudaMemcpy(d_buffer, reinterpret_cast<void*>(data_ptr), width * height * numChannels * sizeof(float), cudaMemcpyDeviceToDevice));

        // Prepare the PBO to display the data in the OpenGL texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        GLenum format = isColor ? GL_RGB : GL_LUMINANCE;
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_FLOAT, NULL);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Clear the OpenGL buffer and render the texture as a quad
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);  // Bottom-left
            glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);   // Bottom-right
            glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);    // Top-right
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);   // Top-left
        glEnd();

        // Swap OpenGL buffers and refresh
        glfwSwapBuffers(window);
        refresh();

        return true;
    }

    // Method to poll for window events (e.g., close, resize)
    void refresh() {
        glfwPollEvents();
    }

    // Method to close the window and release all resources
    bool close() {
        std::lock_guard<std::mutex> lock(glfwMutex);  // Ensure thread-safe closing

        // Unmap and unregister the CUDA resource
        if (cudaResource) {
            CUDA_ASSERT(cudaGraphicsUnmapResources(1, &cudaResource, 0));
            CUDA_ASSERT(cudaGraphicsUnregisterResource(cudaResource));
            cudaResource = nullptr;
        }

        // Destroy the GLFW window and decrement the window count
        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
            windowCount--;
        }

        // Delete the PBO and OpenGL texture
        if (pbo) {
            glDeleteBuffers(1, &pbo);
            pbo = 0;
        }
        if (texture) {
            glDeleteTextures(1, &texture);
            texture = 0;
        }

        // Terminate GLFW only if all windows have been closed
        if (glfwInitialized && windowCount == 0) {
            glfwTerminate();
            glfwInitialized = false;
        }

        return true;
    }

private:
    int width, height;                  // Window dimensions
    float scaleFactor;                  // Scale factor for the window size
    bool isColor;                       // Indicates if the window is for color display
    GLFWwindow* window;                 // GLFW window object
    GLuint pbo, texture;                // PBO and texture handles
    cudaGraphicsResource_t cudaResource;  // CUDA-registered resource for the PBO
    float* d_buffer;                    // Pointer to GPU buffer mapped with CUDA

    // Static variables for GLFW management
    static bool glfwInitialized;        // Track if GLFW has been initialized
    static int windowCount;             // Track the number of active windows
    static std::mutex glfwMutex;        // Mutex to ensure thread-safe access to GLFW
};

// Initialize static variables
bool DisplayWindow::glfwInitialized = false;
int DisplayWindow::windowCount = 0;
std::mutex DisplayWindow::glfwMutex;

// Expose the DisplayWindow class to Python using Pybind11
PYBIND11_MODULE(libDisplay, m) {
    pybind11::class_<DisplayWindow>(m, "DisplayWindow")
        .def(pybind11::init<int, int, float, bool>())  // Constructor: width, height, scaleFactor, isColor
        .def("initialize", &DisplayWindow::initialize) // Initialize the window and resources
        .def("update", &DisplayWindow::update)         // Update the window with new tensor data
        .def("refresh", &DisplayWindow::refresh)       // Poll for window events (prevent "program not responding")
        .def("close", &DisplayWindow::close);          // Close the window and free resources
}
