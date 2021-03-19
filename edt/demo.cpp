/*
Author: Fulin Liu 

File Name: demo.cpp

============================================================================
MIT License

Copyright(c) 2021 Fulin Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this softwareand associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright noticeand this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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
    int maxWidth = 2560;
    int maxHeight = 1280;

    size_t maxLen = maxWidth * maxHeight;
    char* input_dev, * input_host;
    cudaMalloc(&input_dev, maxLen);
    cudaMallocHost(&input_host, maxLen);
    edt::idx2_t* closest_idx_dev;
    cudaMalloc(&closest_idx_dev, maxLen * sizeof(edt::idx2_t));

    float* dist_dev, * dist_host;

    cudaMalloc(&dist_dev, maxLen * sizeof(float));
    cudaMallocHost(&dist_host, maxLen * sizeof(float));

    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t eBegin, eEnd;
    cudaEventCreate(&eBegin);
    cudaEventCreate(&eEnd);
    edt dt;
    dt.setup(maxWidth, maxHeight);
    for (int i = 0; i < 100; ++i) {
        int width = e() % (maxWidth - 1) + 1;
        int height = e() % (maxHeight - 1) + 1;
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
        cudaMemcpyAsync(dist_host, dist_dev, cpyLen * sizeof(float), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        float ms;
        cudaEventElapsedTime(&ms, eBegin, eEnd);
        std::cout << "GPU: " << ms << ", Full: " << t.ms() << "\n\n";

        double minDist, maxDist;
        cv::Mat dist(height, width, CV_32FC1, dist_host);
        cv::minMaxIdx(dist, &minDist, &maxDist);
        cv::Mat normedDist = dist / maxDist;
        cv::imshow("res", normedDist);
        int key = cv::waitKey(0);
        if (key == 27)
            break;
    }
    return 0;
}





