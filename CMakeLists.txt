cmake_minimum_required(VERSION 3.18)
# Force use of GCC 11 if available
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
set(CUDAHOSTCXX /usr/bin/g++-11)


project(SimpleImageFilter LANGUAGES CXX CUDA)



# Enable all warnings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} src)

# Source files
set(SOURCES
    main.cpp
    src/filter_cpu.cpp
    src/sobel_cpu.cpp
    src/filter_cuda.cu
    src/sobel_kernel.cu
)

# Create executable
add_executable(image_filter ${SOURCES})

# Link OpenCV and CUDA
target_link_libraries(image_filter ${OpenCV_LIBS})
