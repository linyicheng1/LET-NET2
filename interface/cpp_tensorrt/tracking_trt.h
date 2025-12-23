#ifndef __TRACKING_TRT_H_
#define __TRACKING_TRT_H_

#include <vector>
#include <string>
#include <memory>
#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include "cuda_runtime_api.h"

class corner_tracking_trt
{
public:
    corner_tracking_trt();
    ~corner_tracking_trt();
    
    // Initialize TensorRT engine
    bool load_model(const std::string& engine_path);
    
    // Main update function with image input
    void update(const cv:: Mat& image);
    
    // Update with gray image and descriptor
    void update(const cv::Mat& gray, const cv::Mat& desc);
    
    // Draw tracking on image (modifies in-place)
    void show(cv::Mat& img);
    
    // Draw tracking and return new image
    cv::Mat draw_tracking(const cv::Mat& img);
    
    // Get number of tracked points
    size_t get_num_points() const { return trackedPoints.size(); }
    
    // Get descriptor from network
    cv::Mat get_descriptor(const cv::Mat& image);
    
private:
    // Extract GFTT features from grayscale image
    std::vector<cv::Point2f> extractFeatureGFTT(
        const cv::Mat& gray,
        const std::vector<cv::Point2f>& existing_points = std::vector<cv:: Point2f>());
    
    // TensorRT members
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;
    
    void** buffers_;
    int nbIO_;
    int inputIndex_;
    int outputIndex_;
    
    struct {
        int height;
        int width;
        int channels;
    } inputDims_;
    
    struct {
        int height;
        int width;
        int channels;
    } outputDims_;
    
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    
    bool model_loaded_;
    
    // Tracking members
    std::vector<cv:: Point2f> trackedPoints;
    std::vector<std:: vector<cv::Point2f>> trackedPointsHistory;
    cv::Mat prevDesc;
    
    // GFTT parameters
    int max_corners_;
    double quality_level_;
    double min_distance_;
    int block_size_;
};

#endif //__TRACKING_TRT_H_