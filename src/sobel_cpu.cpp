#include <opencv2/opencv.hpp>
#include <cmath>

cv::Mat apply_cpu_sobel(const cv::Mat& input_gray) 
{
    CV_Assert(input_gray.type() == CV_8UC1);

    int width = input_gray.cols;
    int height = input_gray.rows;

    cv::Mat output(height, width, CV_8UC1, cv::Scalar(0)); // initialize to black

    for (int y = 1; y < height - 1; ++y) 
    {
        for (int x = 1; x < width - 1; ++x) 
        {
            int gx =
                -1 * input_gray.at<uchar>(y - 1, x - 1) +
                 0 * input_gray.at<uchar>(y - 1, x) +
                 1 * input_gray.at<uchar>(y - 1, x + 1) +
                -2 * input_gray.at<uchar>(y, x - 1) +
                 0 * input_gray.at<uchar>(y, x) +
                 2 * input_gray.at<uchar>(y, x + 1) +
                -1 * input_gray.at<uchar>(y + 1, x - 1) +
                 0 * input_gray.at<uchar>(y + 1, x) +
                 1 * input_gray.at<uchar>(y + 1, x + 1);

            int gy =
                -1 * input_gray.at<uchar>(y - 1, x - 1) +
                -2 * input_gray.at<uchar>(y - 1, x) +
                -1 * input_gray.at<uchar>(y - 1, x + 1) +
                 0 * input_gray.at<uchar>(y, x - 1) +
                 0 * input_gray.at<uchar>(y, x) +
                 0 * input_gray.at<uchar>(y, x + 1) +
                 1 * input_gray.at<uchar>(y + 1, x - 1) +
                 2 * input_gray.at<uchar>(y + 1, x) +
                 1 * input_gray.at<uchar>(y + 1, x + 1);

            int mag = std::abs(gx) + std::abs(gy); // approximation of gradient magnitude
            mag = std::min(255, mag);

            output.at<uchar>(y, x) = static_cast<uchar>(mag);
        }
    }

    return output;
}
