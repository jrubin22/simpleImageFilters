#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

// Forward declarations
cv::Mat apply_cpu_grayscale(const cv::Mat& input);
cv::Mat apply_gpu_grayscale(const cv::Mat& input);
cv::Mat apply_cpu_sobel(const cv::Mat& input_gray);
cv::Mat apply_gpu_sobel(const cv::Mat& input_gray);

int main(int argc, char** argv) 
{
    if (argc < 4) 
    {
        std::cerr << "Usage: " << argv[0] << " <input.jpg> <output.jpg> <mode: cpu|gpu|sobel-cpu|sobel-gpu>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    std::string mode = argv[3];

    // Load image
    cv::Mat input_color = cv::imread(input_path, cv::IMREAD_COLOR);
    if (input_color.empty()) 
    {
        std::cerr << "Error reading image: " << input_path << "\n";
        return 1;
    }

    cv::Mat output;
    auto start = std::chrono::steady_clock::now();

    if (mode == "cpu") 
    {
        std::cout << "Running CPU grayscale filter...\n";
        output = apply_cpu_grayscale(input_color);
    } 
    else if (mode == "gpu") 
    {
        std::cout << "Running GPU grayscale filter...\n";
        output = apply_gpu_grayscale(input_color);
    } 
    else if (mode == "sobel-cpu") 
    {
        std::cout << "Running CPU Sobel filter...\n";
        cv::Mat gray;
        cv::cvtColor(input_color, gray, cv::COLOR_BGR2GRAY);
        output = apply_cpu_sobel(gray);
    } 
    else if (mode == "sobel-gpu") 
    {
        std::cout << "Running GPU Sobel filter...\n";
        cv::Mat gray;
        cv::cvtColor(input_color, gray, cv::COLOR_BGR2GRAY);
        output = apply_gpu_sobel(gray);
    } 
    else 
    {
        std::cerr << "Unknown mode: " << mode << " (use 'cpu', 'gpu', 'sobel-cpu', or 'sobel-gpu')\n";
        return 1;
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Host duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    // Save output
    if (!cv::imwrite(output_path, output)) 
    {
        std::cerr << "Failed to write output image to: " << output_path << "\n";
        return 1;
    }

    std::cout << "Output saved to " << output_path << "\n";
    return 0;
}
