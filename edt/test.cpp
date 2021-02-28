#include "edt.h"
#include "SimpleTimer.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <random>

std::default_random_engine e(time(nullptr));

void randPoints(cv::Mat src, float ratio) {
    int ptCnt = int(src.size().area() * ratio);
    for (int i = 0; i < ptCnt; ++i) {
        int r, c;
        do {
            r = e() % src.rows;
            c = e() % src.cols;
        } while (src.at<unsigned char>(r, c) != 0);
        src.at<unsigned char>(r, c) = 255;
    }
}

int main(int argc, char** argv) {
    int maxWidth = 1920;
    int maxHeight = 1080;

    size_t maxLen = maxWidth * maxHeight;
    char* input_dev, * input_host;
    cudaMalloc(&input_dev, maxLen);
    cudaMallocHost(&input_host, maxLen);
    edt::idx2_t* closest_idx_dev, * closest_idx_host;
    cudaMalloc(&closest_idx_dev, maxLen * sizeof(edt::idx2_t));
    cudaMallocHost(&closest_idx_host, maxLen * sizeof(edt::idx2_t));

    float* dist_dev, * dist_host;
    short2* nodes_host;
    cudaMalloc(&dist_dev, maxLen * sizeof(float));
    cudaMallocHost(&dist_host, maxLen * sizeof(float));
    cudaMallocHost(&nodes_host, maxLen * sizeof(short2));

    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t eBegin, eEnd;
    cudaEventCreate(&eBegin);
    cudaEventCreate(&eEnd);
    edt dt;
    dt.setup(maxWidth, maxHeight);
    for (int i = 0; i < 100; ++i) {
        int width = e() % maxWidth;
        int height = e() % maxHeight;
        std::cout << "W: " << width << ", H: " << height << '\n';
        size_t cpyLen = width * height;
        cv::Mat src(height, width, CV_8UC1);
        src = 0;
        randPoints(src, 0.01);
        memcpy(input_host, src.data, cpyLen);

        const int inputStride = width * sizeof(char);
        const int closestStride = width * sizeof(edt::idx2_t);
        const int distStride = width * sizeof(float);

        fl::Timer t;
        cudaMemcpyAsync(input_dev, input_host, cpyLen, cudaMemcpyHostToDevice, st);
        cudaEventRecord(eBegin, st);
        dt.transform(input_dev, inputStride, closest_idx_dev, closestStride, dist_dev, distStride, width, height, st);
        cudaEventRecord(eEnd, st);
        cudaMemcpyAsync(closest_idx_host, closest_idx_dev, cpyLen * sizeof(edt::idx2_t), cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(dist_host, dist_dev, cpyLen * sizeof(float), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        float ms;
        cudaEventElapsedTime(&ms, eBegin, eEnd);
        std::cout << "GPU: " << ms << ", Full: " << t.ms() << "\n\n";

        double minDist, maxDist;
        cv::Mat dist(height, width, CV_32FC1, dist_host);
        cv::minMaxIdx(dist, &minDist, &maxDist);
        cv::Mat normedDist = dist / maxDist;
        cv::Mat idx(height, width, CV_16SC2, closest_idx_host);
        cv::imshow("res", normedDist);
        int key = cv::waitKey(0);
        if (key == 27)
            break;
    }

    cudaFree(closest_idx_dev);
    cudaFree(dist_dev);
    return 0;
}




