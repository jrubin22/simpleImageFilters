#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void grayscale_kernel(unsigned char* input, unsigned char* output, int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    if (x < width && y < height) 
    {
        unsigned char r = input[idx + 2];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 0];
        output[y * width + x] = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);
    }
}

cv::Mat apply_gpu_grayscale(const cv::Mat& input) 
{
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CV_Assert(input.channels() == 3);
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t input_bytes = width * height * channels;
    size_t output_bytes = width * height;

    // Allocate host output
    cv::Mat output(height, width, CV_8UC1);

    // Allocate device memory
    unsigned char *d_input, *d_output;
    err = cudaMalloc(&d_input, input_bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);

    }
    err = cudaMalloc(&d_output, output_bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);

    }
    // Copy input to device
    err = cudaMemcpy(d_input, input.ptr(), input_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);

    }

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);
    cudaEventRecord(start);
    grayscale_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);

    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;


    // Copy output back to host
    err = cudaMemcpy(output.ptr(), d_output, output_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);

    }
    // Free device memory
    err = cudaFree(d_input);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);

    }
    err = cudaFree(d_output);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);

    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return output;
}
