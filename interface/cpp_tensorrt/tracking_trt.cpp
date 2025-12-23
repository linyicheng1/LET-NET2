#include "tracking_trt.h"
#include <fstream>
#include <iostream>
#include <cstring>

// Logger for TensorRT
class Logger :  public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std:: endl;
        }
    }
} gLogger;

corner_tracking_trt::corner_tracking_trt()
    : runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      stream_(nullptr),
      buffers_(nullptr),
      nbIO_(0),
      inputIndex_(-1),
      outputIndex_(-1),
      model_loaded_(false),
      max_corners_(100),
      quality_level_(0.01),
      min_distance_(20),
      block_size_(3)
{
}

corner_tracking_trt::~corner_tracking_trt() {
    if (buffers_) {
        for (int i = 0; i < nbIO_; ++i) {
            if (buffers_[i]) {
                cudaFree(buffers_[i]);
            }
        }
        delete[] buffers_;
    }
    
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    
    if (context_) {
        delete context_;
    }
    
    if (engine_) {
        delete engine_;
    }
    
    if (runtime_) {
        delete runtime_;
    }
}

bool corner_tracking_trt::load_model(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << engine_path << std:: endl;
        return false;
    }
    
    file.seekg(0, std:: ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();
    
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) {
        std::cerr << "Failed to create runtime" << std::endl;
        return false;
    }
    
    engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }
    
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "Failed to create execution context" << std:: endl;
        return false;
    }
    
    cudaStreamCreate(&stream_);
    
    nbIO_ = engine_->getNbIOTensors();
    buffers_ = new void*[nbIO_];
    
    for (int i = 0; i < nbIO_; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);
        
        size_t vol = 1;
        for (int j = 0; j < dims. nbDims; ++j) {
            vol *= dims.d[j];
        }
        
        size_t bytes = vol * sizeof(float);
        cudaMalloc(&buffers_[i], bytes);
        
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputIndex_ = i;
            // 格式是 NHWC:   [1, H, W, C]
            if (dims.nbDims == 4) {
                inputDims_.height = dims.d[1];    // H = 480
                inputDims_. width = dims.d[2];     // W = 640
                inputDims_.channels = dims. d[3];  // C = 3
            }
            
            std::cout << "Input:  " << name << " [" << dims.d[0] << ", " 
                     << dims.d[1] << ", " << dims.d[2] << ", " << dims. d[3] << "]" << std::endl;
            std::cout << "  Parsed as NHWC: H=" << inputDims_.height 
                     << ", W=" << inputDims_.width 
                     << ", C=" << inputDims_.channels << std:: endl;
        } else {
            outputIndex_ = i;
            // 输出格式是 NHWC:  [1, H, W, C]
            if (dims.nbDims == 4) {
                outputDims_.height = dims.d[1];    // H = 480
                outputDims_.width = dims.d[2];     // W = 640
                outputDims_.channels = dims.d[3];  // C = 4
            }
            
            std::cout << "Output: " << name << " [" << dims. d[0] << ", " 
                     << dims.d[1] << ", " << dims. d[2] << ", " << dims.d[3] << "]" << std::endl;
            std::cout << "  Parsed as NHWC:  H=" << outputDims_.height 
                     << ", W=" << outputDims_.width 
                     << ", C=" << outputDims_.channels << std::endl;
        }
    }
    
    model_loaded_ = true;
    std::cout << "TensorRT model loaded successfully" << std::endl;
    return true;
}

cv::Mat corner_tracking_trt::get_descriptor(const cv:: Mat& image) {
    if (!model_loaded_) {
        std::cerr << "Model not loaded!" << std::endl;
        return cv::Mat();
    }
    
    if (image. empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return cv::Mat();
    }
    
    int H = image.rows;
    int W = image.cols;
    
    if (H != inputDims_.height || W != inputDims_.width) {
        std::cerr << "Input size mismatch:  expected " << inputDims_.height << "x" << inputDims_.width
                  << ", got " << H << "x" << W << std::endl;
        return cv::Mat();
    }
    
    // 准备输入：BGR -> RGB，转为float32（不归一化，模型内部会做）
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3);
    
    // 输入已经是HWC格式，直接使用
    input_buffer_.resize(H * W * 3);
    std::memcpy(input_buffer_.data(), rgb.ptr<float>(), H * W * 3 * sizeof(float));
    
    // Copy to GPU
    cudaMemcpyAsync(buffers_[inputIndex_], input_buffer_.data(), 
                    input_buffer_.size() * sizeof(float), 
                    cudaMemcpyHostToDevice, stream_);
    
    // Set tensor addresses
    for (int i = 0; i < nbIO_; ++i) {
        context_->setTensorAddress(engine_->getIOTensorName(i), buffers_[i]);
    }
    
    // Run inference
    if (! context_->enqueueV3(stream_)) {
        std::cerr << "Inference failed!" << std::endl;
        return cv::Mat();
    }
    
    cudaStreamSynchronize(stream_);
    
    // Get output (NHWC format:  1, H, W, 4)
    int outH = outputDims_.height;
    int outW = outputDims_.width;
    int outC = outputDims_.channels;
    
    size_t outSize = outH * outW * outC;
    output_buffer_.resize(outSize);
    
    cudaMemcpy(output_buffer_.data(), buffers_[outputIndex_], 
               outSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 输出是NHWC格式，直接使用
    // 使用通道 1, 2, 3（跳过通道 0）
    cv::Mat descriptor(outH, outW, CV_8UC3);
    
    for (int h = 0; h < outH; ++h) {
        for (int w = 0; w < outW; ++w) {
            // NHWC格式索引:  (h * W + w) * C + c
            int base_idx = (h * outW + w) * outC;
            
            descriptor.at<cv::Vec3b>(h, w)[0] = cv::saturate_cast<uchar>(output_buffer_[base_idx + 1]);
            descriptor.at<cv::Vec3b>(h, w)[1] = cv::saturate_cast<uchar>(output_buffer_[base_idx + 2]);
            descriptor.at<cv::Vec3b>(h, w)[2] = cv::saturate_cast<uchar>(output_buffer_[base_idx + 3]);
        }
    }
    
    return descriptor;
}

std::vector<cv::Point2f> corner_tracking_trt:: extractFeatureGFTT(
    const cv::Mat& gray,
    const std::vector<cv::Point2f>& existing_points)
{
    if (gray.empty()) {
        return std::vector<cv::Point2f>();
    }
    
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1) * 255;
    
    for (const auto& pt : existing_points) {
        cv::circle(mask, pt, 20, cv::Scalar(0), -1);
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
        0.1
    );
    
    return corners;
}

void corner_tracking_trt::update(const cv::Mat& image) {
    if (!model_loaded_) {
        std::cerr << "Model not loaded!" << std:: endl;
        return;
    }
    
    cv::Mat gray;
    cv::cvtColor(image, gray, cv:: COLOR_BGR2GRAY);
    
    cv::Mat desc = get_descriptor(image);
    if (desc.empty()) {
        return;
    }
    
    update(gray, desc);
}

void corner_tracking_trt::update(const cv::Mat& gray, const cv::Mat& desc) {
    if (gray.empty() || desc.empty()) {
        std::cerr << "Input images are empty!" << std::endl;
        return;
    }
    
    if (trackedPoints.empty()) {
        trackedPoints = extractFeatureGFTT(gray);
        trackedPointsHistory.resize(trackedPoints.size());
        for (size_t i = 0; i < trackedPoints.size(); ++i) {
            trackedPointsHistory[i]. push_back(trackedPoints[i]);
        }
        std::cout << "First frame: extracted " << trackedPoints.size() << " points" << std::endl;
    } else {
        if (trackedPoints.size() > 0 && ! prevDesc.empty()) {
            std::vector<cv::Point2f> trackedPointsNew;
            std::vector<uchar> status;
            std::vector<float> err;
            
            cv::calcOpticalFlowPyrLK(
                prevDesc, desc, trackedPoints, trackedPointsNew,
                status, err, cv:: Size(21, 21), 3,
                cv:: TermCriteria(cv:: TermCriteria::EPS | cv::TermCriteria:: COUNT, 30, 0.01)
            );
            
            std::vector<cv::Point2f> tracked;
            std::vector<std::vector<cv::Point2f>> trackedHistory;
            
            for (size_t i = 0; i < status.size(); ++i) {
                if (status[i]) {
                    tracked.push_back(trackedPointsNew[i]);
                    trackedPointsHistory[i].push_back(trackedPointsNew[i]);
                    
                    if (trackedPointsHistory[i].size() > 5) {
                        trackedPointsHistory[i].erase(trackedPointsHistory[i]. begin());
                    }
                    trackedHistory.push_back(trackedPointsHistory[i]);
                }
            }
            
            std::vector<cv::Point2f> new_points = extractFeatureGFTT(gray, tracked);
            std::vector<std::vector<cv::Point2f>> add_history(new_points.size());
            for (size_t i = 0; i < new_points.size(); ++i) {
                add_history[i].push_back(new_points[i]);
            }
            
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

void corner_tracking_trt::show(cv::Mat& img) {
    if (img.empty()) {
        return;
    }
    
    for (const auto& p : trackedPoints) {
        cv::circle(img, p, 4, cv::Scalar(0, 255, 0), -1);
        cv::circle(img, p, 6, cv::Scalar(0, 255, 0), 1);
    }
    
    for (const auto& history : trackedPointsHistory) {
        for (size_t i = 1; i < history.size(); ++i) {
            cv::line(img, history[i - 1], history[i], cv::Scalar(0, 0, 255), 2);
        }
    }
    
    std::string text = "Tracked Points: " + std::to_string(trackedPoints.size());
    cv::putText(img, text, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
}

cv::Mat corner_tracking_trt::draw_tracking(const cv::Mat& img) {
    cv::Mat vis = img.clone();
    
    if (vis.empty()) {
        return vis;
    }
    
    for (const auto& p : trackedPoints) {
        cv::circle(vis, p, 4, cv::Scalar(0, 255, 0), -1);
        cv::circle(vis, p, 6, cv:: Scalar(0, 255, 0), 1);
    }
    
    for (const auto& history : trackedPointsHistory) {
        for (size_t i = 1; i < history.size(); ++i) {
            cv::line(vis, history[i - 1], history[i], cv::Scalar(0, 0, 255), 2);
        }
    }
    
    std::string text = "Tracked Points: " + std::to_string(trackedPoints.size());
    cv::putText(vis, text, cv::Point(10, 30), 
                cv:: FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    return vis;
}