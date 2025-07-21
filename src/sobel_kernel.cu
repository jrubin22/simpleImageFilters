#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void sobel_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int input_step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip border pixels
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)
        return;

    int offset = y * input_step + x;

    // Sobel X
    int Gx =
        -1 * input[(y - 1) * input_step + (x - 1)] +
         0 * input[(y - 1) * input_step + (x    )] +
         1 * input[(y - 1) * input_step + (x + 1)] +
        -2 * input[(y    ) * input_step + (x - 1)] +
         0 * input[(y    ) * input_step + (x    )] +
         2 * input[(y    ) * input_step + (x + 1)] +
        -1 * input[(y + 1) * input_step + (x - 1)] +
         0 * input[(y + 1) * input_step + (x    )] +
         1 * input[(y + 1) * input_step + (x + 1)];

    // Sobel Y
    int Gy =
        -1 * input[(y - 1) * input_step + (x - 1)] +
        -2 * input[(y - 1) * input_step + (x    )] +
        -1 * input[(y - 1) * input_step + (x + 1)] +
         0 * input[(y    ) * input_step + (x - 1)] +
         0 * input[(y    ) * input_step + (x    )] +
         0 * input[(y    ) * input_step + (x + 1)] +
         1 * input[(y + 1) * input_step + (x - 1)] +
         2 * input[(y + 1) * input_step + (x    )] +
         1 * input[(y + 1) * input_step + (x + 1)];

    int magnitude = abs(Gx) + abs(Gy);  // approximation

    // Clamp to [0, 255]
    magnitude = min(255, magnitude);

    output[offset] = static_cast<unsigned char>(magnitude);
}


cv::Mat apply_gpu_sobel(const cv::Mat& input_gray) 
{
    CV_Assert(input_gray.type() == CV_8UC1);

    int width = input_gray.cols;
    int height = input_gray.rows;
    size_t input_bytes = input_gray.step * height;
    size_t output_bytes = width * height;

    cv::Mat output(height, width, CV_8UC1);

    unsigned char *d_input = nullptr, *d_output = nullptr;
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory
    err = cudaMalloc(&d_input, input_bytes);
    if (err != cudaSuccess) 
    {
        std::cerr << "cudaMalloc d_input failed: " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_output, output_bytes);
    if (err != cudaSuccess) 
    {
        std::cerr << "cudaMalloc d_output failed: " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Copy input to device
    err = cudaMemcpy(d_input, input_gray.ptr(), input_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);
    cudaEventRecord(start);
    sobel_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, input_gray.step);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) 
    {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;


    // Copy output back
    err = cudaMemcpy(output.ptr(), d_output, output_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) 
    {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
