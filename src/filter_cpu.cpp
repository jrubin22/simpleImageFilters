#include <opencv2/opencv.hpp>

// Convert to grayscale using standard luma transform
cv::Mat apply_cpu_grayscale(const cv::Mat& input) 
{
    CV_Assert(input.channels() == 3);

    cv::Mat gray(input.rows, input.cols, CV_8UC1);

    for (int y = 0; y < input.rows; ++y) 
    {
        for (int x = 0; x < input.cols; ++x) 
        {
            const cv::Vec3b& pixel = input.at<cv::Vec3b>(y, x);
            uint8_t gray_val = static_cast<uint8_t>(0.21f * pixel[2] + 0.72f * pixel[1] + 0.07f * pixel[0]);
            gray.at<uint8_t>(y, x) = gray_val;
        }
    }

    return gray;
}
