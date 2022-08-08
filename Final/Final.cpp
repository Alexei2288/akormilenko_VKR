#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

const char* filename = "C:\\samples\\1.jpg";

Mat src, src_gray;
Mat dst, detected_edges, grad;
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
int scale = 1, delta = 0, ddepth = CV_16S;
const char* window_name = "Canny edge Map";

// Кэнни
static void CannyThreshold(int, void*)
{
    blur(src_gray, detected_edges, Size(3, 3));
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);
    imshow(window_name, dst);
}

// Марра-Хилдрет
void ImageAdjust(cv::Mat& src, cv::Mat& dst) {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int g = src.at<uchar>(i, j);
            dst.at<float>(i, j) = g / 255.;
        }
    }
}

void GaussianKernel(cv::Mat& ss, int size, double delta) {
    int centerX = size / 2;
    int centerY = size / 2;
    int x = 0;
    int y = 0;
    cv::Mat gaussiankernel;
    gaussiankernel = cv::Mat_<double>(size, size);
    double cc = 0.5 / (3.1415926 * delta * delta);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            x = std::abs(i - centerX);
            y = std::abs(j - centerY);
            double dd = cc * exp(-((pow(x, 2) + pow(y, 2)) / (2 * delta * delta)));
            gaussiankernel.at<double>(i, j) = dd;
        }
    }

    cv::Mat ss1;
    ss1 = cv::Mat_<int>(size, size);
    int sum = 0;
    // Преобразование десятичного типа преобразования Гаусса в целое число
    double ff = gaussiankernel.at<double>(0, 0);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int cc = cvRound(gaussiankernel.at<double>(i, j) / ff);
            ss1.at<int>(i, j) = cc;
            sum += cc;
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int cc = ss1.at<int>(i, j);
            ss.at<float>(i, j) = (float)(cc) / sum;
        }
    }
}

void Laplacian(cv::Mat& src, cv::Mat& dst)
{
    cv::Mat element;
    element = (cv::Mat_<int>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            double value = 0;
            int i = 0;
            for (int m = y - 1; m <= y + 1; ++m) {
                for (int n = x - 1; n <= x + 1; ++n) {
                    int m1 = i / 3;
                    int n1 = i % 3;
                    int t = element.at<int>(m1, n1);
                    i++;
                    float b = src.at<float>(m, n);
                    value += t * b;
                }
            }

            dst.at<float>(y, x) = value;
        }
    }
}

void Edjes(cv::Mat& src, cv::Mat& result, double threshold) {
    for (int y = 1; y < src.rows - 1; ++y)
    {
        for (int x = 1; x < src.cols; ++x)
        {
            // Решение о соседстве
            if ((src.at<float>(y - 1, x) *
                src.at<float>(y + 1, x) < 0) && (std::abs(src.at<float>(y - 1, x) -
                    src.at<float>(y + 1, x)) > threshold))
            {
                result.at<uchar>(y, x) = 255;
            }
            if ((src.at<float>(y, x - 1) *
                src.at<float>(y, x + 1) < 0) && (std::abs(src.at<float>(y, x - 1) -
                    src.at<float>(y, x + 1)) > threshold))
            {
                result.at<uchar>(y, x) = 255;
            }
            if ((src.at<float>(y + 1, x - 1) *
                src.at<float>(y - 1, x + 1) < 0) && (std::abs(src.at<float>(y + 1, x - 1) -
                    src.at<float>(y - 1, x + 1)) > threshold))
            {
                result.at<uchar>(y, x) = 255;
            }
            if ((src.at<float>(y - 1, x - 1) *
                src.at<float>(y + 1, x + 1) < 0) && (std::abs(src.at<float>(y - 1, x - 1) -
                    src.at<float>(y + 1, x + 1)) > threshold))
            {
                result.at<uchar>(y, x) = 255;
            }
        }
    }
}

void Marra_Hildred(cv::Mat& src)
{
    cv::Mat gray(src.size(), CV_32FC1);
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    /*cv::imshow("gray", gray);
    cv::imwrite("../gray.png", gray);
    waitKey(0);*/

    cv::Mat gray01;
    gray01 = cv::Mat::zeros(gray.size(), CV_32FC1);
    ImageAdjust(gray, gray01);
    /*cv::imshow("gray01", gray01);
    waitKey(0);*/

    int size = 13; // Размер матрицы ядра Гаусса
    double delta = 2;
    cv::Mat gaussiankernel;
    cv::Mat gaussianblur;
    gaussiankernel = cv::Mat_<float>(size, size);//(cv::Size(size,size),CV_64FC1);
    GaussianKernel(gaussiankernel, size, delta);
    cv::filter2D(gray01, gaussianblur, -1, gaussiankernel);

    cv::Mat Laplace;
    Laplace = cv::Mat::zeros(gaussianblur.size(), CV_32FC1);
    Laplacian(gaussianblur, Laplace);
    /*cv::imshow("Laplace", Laplace);
    waitKey(0);*/

    double max = 0;
    double min = 0;
    cv::minMaxLoc(Laplace, &min, &max);

    cv::Mat result;
    result = cv::Mat::zeros(Laplace.size(), CV_8U);
    Edjes(Laplace, result, 0.02);
    cv::imshow("Marra-Hildred", result);

}

// Собель
void Sobell(cv::Mat& src)
{
    /// Применяем размытие по Гауссу
    GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

    /// Конвертируем его
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    /// Объявляем градиенты grad_x и grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Градиент по X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x); // модуль значения градиента 

    /// Градиент по Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y); // модуль значения градиента 

    /// Слияние градиентов (аппроксимация)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    imshow("Sobell", grad);

    waitKey(0);
}




int main(int argc, char** argv)
{
    src = imread(filename, IMREAD_COLOR);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }


    // Оператор Кэнни
    dst.create(src.size(), src.type());
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    namedWindow(window_name, WINDOW_AUTOSIZE);
    createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
    CannyThreshold(0, 0);
    waitKey(0);

    // Оператор Марра-Хилдрет 
    src = imread(filename, IMREAD_COLOR);
    Marra_Hildred(src);
    waitKey(0);

    // Оператор Собеля
    src = imread(filename, IMREAD_COLOR);
    Sobell(src);
    waitKey(0);


    return 0;
}