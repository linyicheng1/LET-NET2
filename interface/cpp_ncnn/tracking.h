#ifndef __TRACKING_H_
#define __TRACKING_H_
#include <vector>
#include "opencv2/opencv.hpp"
#include "net.h"  // ncnn

class corner_tracking
{
public:
    corner_tracking();
    ~corner_tracking() = default;
    
    // Initialize ncnn model
    bool load_model(const std::string& param_path, const std::string& bin_path);
    
    // Main update function with image input
    void update(const cv:: Mat& image);
    
    // Update with gray image and descriptor
    void update(const cv::Mat& gray, const cv::Mat& desc);
    
    // Draw tracking on image (modifies in-place)
    void show(cv::Mat& img);
    
    // Draw tracking and return new image
    cv::Mat draw_tracking(const cv::Mat& img);
    
    // Get number of tracked points
    size_t get_num_points() const { return trackedPoints. size(); }
    
    // Get descriptor from network
    cv::Mat get_descriptor(const cv::Mat& image);
    
private:
    // Extract GFTT features from grayscale image
    std::vector<cv::Point2f> extractFeatureGFTT(
        const cv::Mat& gray,
        const std::vector<cv::Point2f>& existing_points = std::vector<cv:: Point2f>());

    std::vector<cv::Point2f> trackedPoints;
    std::vector<std::vector<cv::Point2f>> trackedPointsHistory;
    cv::Mat prevDesc;
    
    // ncnn network
    ncnn:: Net net_;
    bool model_loaded_;
    
    // GFTT parameters
    int max_corners_;
    double quality_level_;
    double min_distance_;
    int block_size_;
};

#endif //__TRACKING_H_