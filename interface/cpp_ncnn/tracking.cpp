#include "tracking.h"

corner_tracking::corner_tracking() 
    : model_loaded_(false),
      max_corners_(200),
      quality_level_(0.01),
      min_distance_(10),
      block_size_(3)
{
}

bool corner_tracking::load_model(const std::string& param_path, const std::string& bin_path) {
    // Set ncnn options
    net_.opt.num_threads = 4;
    net_.opt.use_vulkan_compute = false;
    
    // Load ncnn model
    int ret_param = net_.load_param(param_path.c_str());
    int ret_model = net_.load_model(bin_path.c_str());
    
    if (ret_param != 0) {
        fprintf(stderr, "Failed to load param file: %s (error code: %d)\n", param_path.c_str(), ret_param);
        return false;
    }
    
    if (ret_model != 0) {
        fprintf(stderr, "Failed to load model file: %s (error code: %d)\n", bin_path.c_str(), ret_model);
        return false;
    }
    
    model_loaded_ = true;
    printf("Model loaded successfully with %d threads\n", net_.opt.num_threads);
    return true;
}

cv::Mat corner_tracking::get_descriptor(const cv::Mat& image) {
    if (!model_loaded_) {
        fprintf(stderr, "Model not loaded!\n");
        return cv::Mat();
    }
    
    if (image.empty()) {
        fprintf(stderr, "Input image is empty!\n");
        return cv::Mat();
    }
    
    // 输入BGR格式
    cv::Mat bgr_image = image.clone();
    
    // 创建ncnn输入
    ncnn::Mat in = ncnn::Mat::from_pixels(
        bgr_image.data, 
        ncnn::Mat::PIXEL_BGR, 
        bgr_image.cols, 
        bgr_image.rows
    );
    
    // Forward inference
    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(true);
    
    // 使用正确的输入输出名称
    int ret_input = ex.input("in0", in);
    if (ret_input != 0) {
        fprintf(stderr, "Failed to set input (error code: %d)\n", ret_input);
        return cv::Mat();
    }
    
    ncnn::Mat out;
    int ret_extract = ex.extract("out0", out);
    if (ret_extract != 0) {
        fprintf(stderr, "Failed to extract output (error code: %d)\n", ret_extract);
        return cv::Mat();
    }
    
    // 输出shape:  (4, H, W)
    int channels = out.c;
    int height = out. h;
    int width = out.w;
    
    // 创建输出Mat，使用后3个通道
    cv::Mat descriptor(height, width, CV_8UC3);
    
    // 转换ncnn:: Mat到cv::Mat
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            // 使用通道1, 2, 3
            const float* c1 = out.channel(1);
            const float* c2 = out.channel(2);
            const float* c3 = out.channel(3);
            
            int idx = h * width + w;
            
            descriptor.at<cv::Vec3b>(h, w)[0] = cv::saturate_cast<uchar>(c1[idx]);
            descriptor.at<cv::Vec3b>(h, w)[1] = cv::saturate_cast<uchar>(c2[idx]);
            descriptor.at<cv::Vec3b>(h, w)[2] = cv::saturate_cast<uchar>(c3[idx]);
        }
    }
    
    return descriptor;
}

std::vector<cv::Point2f> corner_tracking::extractFeatureGFTT(
    const cv::Mat& gray,
    const std::vector<cv::Point2f>& existing_points)
{
    if (gray.empty()) {
        return std::vector<cv::Point2f>();
    }
    
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1) * 255;
    
    for (const auto& pt : existing_points) {
        cv::circle(mask, pt, 10, cv::Scalar(0), -1);
    }
    
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
        gray,
        corners,
        max_corners_,
        quality_level_,
        min_distance_,
        mask,
        block_size_,
        false,
        0.04
    );
    
    return corners;
}

void corner_tracking::update(const cv::Mat& image) {
    if (!model_loaded_) {
        fprintf(stderr, "Model not loaded!  Call load_model() first.\n");
        return;
    }
    
    // 转换为灰度图
    cv::Mat gray;
    cv:: cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // 获取descriptor
    cv::Mat desc = get_descriptor(image);
    
    if (desc.empty()) {
        return;
    }
    
    // 调用重载的update函数
    update(gray, desc);
}

void corner_tracking::update(const cv::Mat& gray, const cv:: Mat& desc) {
    if (gray.empty() || desc.empty()) {
        fprintf(stderr, "Input images are empty!\n");
        return;
    }
    
    if (trackedPoints.empty()) {
        // 第一帧：提取GFTT特征
        trackedPoints = extractFeatureGFTT(gray);
        trackedPointsHistory.resize(trackedPoints.size());
        for (size_t i = 0; i < trackedPoints.size(); i++) {
            trackedPointsHistory[i].push_back(trackedPoints[i]);
        }
        printf("First frame:  extracted %zu points\n", trackedPoints.size());
    } else {
        // 使用光流跟踪
        if (trackedPoints.size() > 0 && ! prevDesc.empty()) {
            std::vector<cv::Point2f> trackedPointsNew;
            std::vector<uchar> status;
            std::vector<float> err;

            cv::calcOpticalFlowPyrLK(
                prevDesc,
                desc,
                trackedPoints,
                trackedPointsNew,
                status,
                err,
                cv::Size(21, 21),
                3,
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01)
            );

            std::vector<cv::Point2f> tracked;
            std::vector<std::vector<cv::Point2f>> trackedHistory;
            
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i]) {
                    tracked.push_back(trackedPointsNew[i]);
                    trackedPointsHistory[i].push_back(trackedPointsNew[i]);
                    
                    if (trackedPointsHistory[i].size() > 5) {
                        trackedPointsHistory[i].erase(trackedPointsHistory[i]. begin());
                    }
                    trackedHistory.push_back(trackedPointsHistory[i]);
                }
            }
            
            // 提取新特征
            std::vector<cv::Point2f> new_points = extractFeatureGFTT(gray, tracked);
            
            std::vector<std::vector<cv::Point2f>> add_history(new_points.size());
            for (size_t i = 0; i < new_points.size(); i++) {
                add_history[i].push_back(new_points[i]);
            }
            
            // 合并
            trackedPoints. clear();
            trackedPointsHistory.clear();

            trackedPoints.insert(trackedPoints.end(), tracked.begin(), tracked.end());
            trackedPoints.insert(trackedPoints.end(), new_points.begin(), new_points.end());
            trackedPointsHistory.insert(trackedPointsHistory.end(), trackedHistory.begin(), trackedHistory.end());
            trackedPointsHistory.insert(trackedPointsHistory.end(), add_history.begin(), add_history.end());
        }
    }
    
    prevDesc = desc. clone();
}

void corner_tracking::show(cv::Mat &img) {
    if (img.empty()) {
        fprintf(stderr, "Input image to show() is empty!\n");
        return;
    }
    
    // 绘制跟踪点（绿色）
    for (const auto& p : trackedPoints) {
        cv::circle(img, p, 4, cv::Scalar(0, 255, 0), -1);
        cv::circle(img, p, 6, cv::Scalar(0, 255, 0), 1);  // 外圈
    }
    
    // 绘制轨迹（红色）
    for (const auto& history : trackedPointsHistory) {
        for (size_t i = 1; i < history.size(); i++) {
            cv::line(img, history[i - 1], history[i], cv::Scalar(0, 0, 255), 2);
        }
    }
    
    // 显示点数
    std::string text = "Tracked Points: " + std::to_string(trackedPoints.size());
    cv::putText(img, text, cv:: Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv:: Scalar(0, 255, 0), 2);
}

cv::Mat corner_tracking:: draw_tracking(const cv::Mat& img) {
    cv::Mat vis = img.clone();
    
    if (vis.empty()) {
        fprintf(stderr, "Input image is empty!\n");
        return vis;
    }
    
    // 绘制跟踪点（绿色，双圈）
    for (const auto& p : trackedPoints) {
        cv::circle(vis, p, 4, cv:: Scalar(0, 255, 0), -1);  // 实心圆
        cv::circle(vis, p, 6, cv::Scalar(0, 255, 0), 1);   // 外圈
    }
    
    // 绘制轨迹（红色）
    for (const auto& history : trackedPointsHistory) {
        for (size_t i = 1; i < history. size(); i++) {
            cv::line(vis, history[i - 1], history[i], cv::Scalar(0, 0, 255), 2);
        }
    }
    
    // 显示信息
    std::string text = "Tracked Points: " + std::to_string(trackedPoints.size());
    cv::putText(vis, text, cv:: Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv:: Scalar(0, 255, 0), 2);
    
    return vis;
}