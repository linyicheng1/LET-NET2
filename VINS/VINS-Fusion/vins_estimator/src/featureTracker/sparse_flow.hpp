#ifndef SPARSE_FLOW_HPP
#define SPARSE_FLOW_HPP

#include <opencv2/opencv.hpp>
#include <thread>
#include <omp.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <opencv2/opencv.hpp>



class SparseFlow {
public:
    explicit SparseFlow(int patch_size);

    std::pair<cv::Mat, cv::Mat> encoder_image(cv::Mat &input_image);

    void calcOpticalFlow(
        const std::pair<cv::Mat, cv::Mat>& pair_feat0,
        const std::pair<cv::Mat, cv::Mat>& pair_feats1,
        const std::vector<cv::Point2f>& points0,
        std::vector<cv::Point2f>& points1,
        std::vector<uchar>& status,
        std::vector<float>& error) const;

    static cv::Mat makeHWGrid(int patchSize);

    static std::vector<cv::Mat> extractFeatureVectorsAtCorners(
        const cv::Mat& featureMap,                       // CV_32FC(N)
        const std::vector<cv::Point2f>& corners,         // 原图角点
        float scaleFactor);                               // 特征图相对原图的缩放因子

    static std::vector<std::pair<cv::Mat, cv::Point2f>> extractFeaturePatchesAtCornersWithCoords(
        const cv::Mat& featureMap,                       // CV_32FC(N)
        const std::vector<cv::Point2f>& corners,         // 原图角点
        float scaleFactor,                               // 缩放因子（如 1.0/8）
        int patchSize);                                  // Patch 尺寸（奇数，例如 5 表示 5x5）

    static std::pair<std::vector<cv::Point2f>, std::vector<float>> computeFlowAndScoreFromFeatures_MatWithHWGrid (
        const std::vector<cv::Mat>& features,
        const std::vector<std::pair<cv::Mat, cv::Point2f>>& patches,
        const cv::Mat& hw_grid,   // (H*W, 2), CV_32F
        int patchSize,
        float scale,
        float temperature = 0.02f);

    static std::pair<std::vector<cv::Point2f>, std::vector<float>>computeFlowAndScoreFromFeaturesAndPatches(
        const std::vector<cv::Mat>& features,   // 每个点的 1xC 特征向量
        const std::vector<std::pair<cv::Mat, cv::Point2f>>& patches,    // 每个点的 patchSize x patchSize x C 特征 patch
        int patchSize,                           // patch 边长（奇数）
        float scale);

    static std::pair<std::vector<cv::Point2f>, std::vector<float>>computeFlowAndScoreSubPix(
        const std::vector<cv::Mat>& features,   // 每个点的 1xC 特征向量
        const std::vector<std::pair<cv::Mat, cv::Point2f>>& patches,    // 每个点的 patchSize x patchSize x C 特征 patch
        int patchSize,                           // patch 边长（奇数）
        float scale);
private:
    int patchSize = 21;
    float scoreThreshold = 50.0f;
    cv::Mat hw_grid;
};


using namespace nvinfer1;

struct TRTOutput {
    float* data;  // 直接设备或 host 数据指针
    int C, H, W;
};

class TRTInferV3 {
public:
    TRTInferV3(const std::string& enginePath);
    ~TRTInferV3();

    cv::Mat infer(const cv::Mat& input);  // 接受 preprocessed host input
    void infer(const cv::Mat& input, std::vector<float> &feat0, std::vector<float> &feat1, std::vector<float> &feat2,
               std::vector<float> &con0, std::vector<float> &con1, std::vector<float> &con2);
    cv::Size getInputDims() const { return inputDims; }

private:
    void initBindings();
    IRuntime* runtime{nullptr};
    ICudaEngine* engine{nullptr};
    IExecutionContext* context{nullptr};
    cudaStream_t stream;
    std::vector<void*> buffers;
    int nbIO = 0, inputIndex = 0;
    cv::Size inputDims;

    std::vector<float> feat_buffer;
};

class SparsFlowTRT {
public:
    SparsFlowTRT(const std::string& engine_path);
    ~SparsFlowTRT();

    void infer(const std::vector<float>& feat00, const std::vector<float>& feat10, const std::vector<float>& con1,
               const std::vector<float>& feat01, const std::vector<float>& feat11, const std::vector<float>& con2,
               const std::vector<float>& feat02, const std::vector<float>& feat12, const std::vector<float>& con3,
               const std::vector<cv::Point2f>& pts0, const std::vector<cv::Point2f>& pts1,
               std::vector<cv::Point2f>& out_pts2, std::vector<uchar>& out_scores);


private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_{};

    std::vector<void*> buffers_;
    std::vector<nvinfer1::Dims> dims_;
    std::vector<bool> is_input_;

    void loadEngine(const std::string& path);
    void allocateBuffers();
};



class LKFlowInfer {
public:
    explicit LKFlowInfer(const std::string& enginePath);
    ~LKFlowInfer();
    void infer(const cv::Mat& input, std::vector<cv::Mat> &feats);
    int build_pyr(cv::InputArrayOfArrays _pyramidImages,
        cv::OutputArrayOfArrays pyramid,
        cv::Size winSize = cv::Size(21, 21),
        int maxLevel = 3,
        bool withDerivatives = cv::BORDER_REFLECT_101,
        int pyrBorder = cv::BORDER_CONSTANT,
        int derivBorder = true);

private:
    void initBindings();
    IRuntime* runtime{nullptr};
    ICudaEngine* engine{nullptr};
    IExecutionContext* context{nullptr};
    cudaStream_t stream{};
    std::vector<void*> buffers;
    int nbIO = 0, inputIndex = 0;
    cv::Size inputDims;

    std::vector<float> feat_buffer;
};


class LETFlowInfer {
public:
    explicit LETFlowInfer(const std::string& enginePath);
    ~LETFlowInfer();

    cv::Mat infer(const cv::Mat& input);
    void infer2(const cv::Mat& input, cv::Mat& cov, cv::Mat& desc);

private:
    void initBindings();
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;

    std::vector<void*> buffers;
    int nbIO = 0;
    int inputIndex = -1;
    cv::Size inputDims;

    std::vector<float> output_buffer;
    std::vector<float> output_buffer2;
};


#endif //SPARSE_FLOW_HPP
