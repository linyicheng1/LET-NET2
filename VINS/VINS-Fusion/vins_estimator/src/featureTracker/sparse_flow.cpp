#include "sparse_flow.hpp"
#include <fstream>
#include <iostream>
#define PTS_NUM 250


SparseFlow::SparseFlow(int patch_size)
    :patchSize(patch_size) {
    hw_grid = makeHWGrid(patchSize);
}


void SparseFlow::calcOpticalFlow(const std::pair<cv::Mat, cv::Mat> &pair_feat0, const std::pair<cv::Mat, cv::Mat> &pair_feats1, const std::vector<cv::Point2f> &points0, std::vector<cv::Point2f> &points1, std::vector<uchar> &status, std::vector<float> &error) const {
    auto [feat0, feat1] = pair_feat0;
    auto [feat0b, feat1b] = pair_feats1;
    // Step 2: 特征提取
    auto feats0 = extractFeatureVectorsAtCorners(feat1, points0, 1.0f / 8);
    auto patches0 = extractFeaturePatchesAtCornersWithCoords(feat1b, points0, 1.0f / 8, patchSize);

    // Step 3: 匹配与估计
    auto [flow, scores] = computeFlowAndScoreSubPix(feats0, patches0, patchSize, 8.0f);

    // auto feats1 = extractFeatureVectorsAtCorners(feat0, points0, 1.0f / 2);
    // auto patches1 = extractFeaturePatchesAtCornersWithCoords(feat0b, flow, 1.0f / 2, patchSize);
    //
    // auto [flow1, scores1] = computeFlowAndScoreSubPix(feats1, patches1, patchSize, 2.0f);


    // Step 4: 输出格式化
    points1.clear();
    status.clear();
    error.clear();

    for (size_t i = 0; i < points0.size(); ++i) {
        points1.push_back(flow[i]);
        status.push_back(scores[i] + scores[i] < scoreThreshold ? 1 : 0);
        error.push_back(scores[i]);
    }
}


cv::Mat SparseFlow::makeHWGrid(int patchSize) {
    cv::Mat hw_grid(patchSize * patchSize, 2, CV_32F);

    int idx = 0;
    for (int y = 0; y < patchSize; ++y) {
        for (int x = 0; x < patchSize; ++x) {
            hw_grid.at<float>(idx, 0) = static_cast<float>(x);  // dx
            hw_grid.at<float>(idx, 1) = static_cast<float>(y);  // dy
            ++idx;
        }
    }
    return hw_grid;  // (H*W, 2)
}

std::vector<cv::Mat> SparseFlow::extractFeatureVectorsAtCorners(const cv::Mat &featureMap, const std::vector<cv::Point2f> &corners, float scaleFactor) {
    const int featH = featureMap.rows;
    const int featW = featureMap.cols;
    const int featDim = featureMap.channels();

    std::vector<cv::Mat> featureVectors(corners.size());

#pragma omp parallel for
    for (int i = 0; i < corners.size(); ++i) {
        const auto& pt = corners[i];

        float fx = pt.x * scaleFactor;
        float fy = pt.y * scaleFactor;

        int x = static_cast<int>(fx);
        int y = static_cast<int>(fy);
        float dx = fx - x;
        float dy = fy - y;

        // 越界检查
        if (x < 0 || x + 1 >= featW || y < 0 || y + 1 >= featH) {
            featureVectors[i] = cv::Mat::zeros(1, featDim, CV_32F);
            continue;
        }

        const float* f00 = featureMap.ptr<float>(y, x);
        const float* f01 = featureMap.ptr<float>(y, x + 1);
        const float* f10 = featureMap.ptr<float>(y + 1, x);
        const float* f11 = featureMap.ptr<float>(y + 1, x + 1);

        cv::Mat feature(1, featDim, CV_32F);
        float* dst = feature.ptr<float>();

        for (int c = 0; c < featDim; ++c) {
            dst[c] =
                (1 - dx) * (1 - dy) * f00[c] +
                dx       * (1 - dy) * f01[c] +
                (1 - dx) * dy       * f10[c] +
                dx       * dy       * f11[c];
        }

        featureVectors[i] = feature;
    }

    return featureVectors;
}


std::vector<std::pair<cv::Mat, cv::Point2f> > SparseFlow::extractFeaturePatchesAtCornersWithCoords(const cv::Mat &featureMap, const std::vector<cv::Point2f> &corners, float scaleFactor, int patchSize) {
    std::vector<std::pair<cv::Mat, cv::Point2f>> patchesWithCoords;
    int halfSize = patchSize / 2;

    // 边缘 padding
    cv::Mat paddedFeatureMap;
    cv::copyMakeBorder(
        featureMap,
        paddedFeatureMap,
        halfSize + halfSize, halfSize + halfSize,
        halfSize + halfSize, halfSize + halfSize,
        cv::BORDER_CONSTANT,
        cv::Scalar::all(0)
    );

    for (const auto& pt : corners) {
        int fx = static_cast<int>(pt.x * scaleFactor + 0.5f);
        int fy = static_cast<int>(pt.y * scaleFactor + 0.5f);

        // 边缘padding偏移
        int x0 = fx + halfSize;
        int y0 = fy + halfSize;

        cv::Rect roi(x0, y0, patchSize, patchSize);

        // 直接引用 ROI，返回patch和左上角坐标
        patchesWithCoords.emplace_back(paddedFeatureMap(roi), cv::Point2f(x0 - halfSize - halfSize, y0 - halfSize - halfSize));
    }

    return patchesWithCoords;
}

std::pair<std::vector<cv::Point2f>, std::vector<float> > SparseFlow::computeFlowAndScoreFromFeatures_MatWithHWGrid(const std::vector<cv::Mat> &features, const std::vector<std::pair<cv::Mat, cv::Point2f> > &patches, const cv::Mat &hw_grid, int patchSize, float scale, float temperature) {
    const int n = static_cast<int>(features.size());
        std::vector<cv::Point2f> flow(n);
        std::vector<float> scores(n);

        const int dim = features[0].cols;
        const int gridLen = patchSize * patchSize;

#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            const auto& feat = features[i];  // 1 x C
            const auto& patch = patches[i].first;  // patchSize x patchSize x C
            const auto& topLeft = patches[i].second;

            if (feat.empty() || patch.empty()) {
                flow[i] = cv::Point2f(0, 0);
                scores[i] = 1e9f;
                continue;
            }

            // Flatten patch to (gridLen x C)
            cv::Mat patch_flat(gridLen, dim, CV_32F);
            for (int y = 0; y < patchSize; ++y) {
                const float* rowPtr = patch.ptr<float>(y);
                for (int x = 0; x < patchSize; ++x) {
                    float* dst = patch_flat.ptr<float>(y * patchSize + x);
                    std::memcpy(dst, rowPtr + x * dim, sizeof(float) * dim);
                }
            }

            // Distance squared
            cv::Mat diff = patch_flat - cv::repeat(feat, gridLen, 1);  // (gridLen x C)
            cv::Mat dist2;
            cv::reduce(diff.mul(diff), dist2, 1, cv::REDUCE_SUM);  // (gridLen x 1)

            // Softmax weights
            cv::Mat logits = -dist2 / temperature;
            double maxVal;
            cv::minMaxLoc(logits, nullptr, &maxVal);
            logits -= maxVal;
            cv::exp(logits, logits);  // (gridLen x 1)
            float denom = static_cast<float>(cv::sum(logits)[0]);

            if (denom < 1e-6f) {
                flow[i] = cv::Point2f(0, 0);
                scores[i] = 1e9f;
                continue;
            }

            cv::Mat weights = logits / denom;  // (gridLen x 1)

            // 计算加权偏移 (gridLen x 2)
            cv::Mat weightedGrid;
            cv::multiply(hw_grid, cv::repeat(weights, 1, 2), weightedGrid);  // (gridLen x 2)

            cv::Mat sumOffset;  // 将得到 (1, 2)
            cv::reduce(weightedGrid, sumOffset, 0, cv::REDUCE_SUM);

            float dx = sumOffset.at<float>(0, 0);
            float dy = sumOffset.at<float>(0, 1);

            flow[i] = (cv::Point2f(dx, dy) + topLeft) * scale;
            scores[i] = denom;
        }

        return {flow, scores};
}


std::pair<std::vector<cv::Point2f>, std::vector<float> > SparseFlow::computeFlowAndScoreFromFeaturesAndPatches(const std::vector<cv::Mat> &features, const std::vector<std::pair<cv::Mat, cv::Point2f> > &patches, int patchSize, float scale) {
    int n = static_cast<int>(features.size());
    std::vector<cv::Point2f> flow(n, cv::Point2f(0, 0));
    std::vector<float> scores(n, std::numeric_limits<float>::max());
    int halfPatch = patchSize / 2;

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (features[i].empty() || patches[i].first.empty()) {
            flow[i] = cv::Point2f(0, 0);
            scores[i] = std::numeric_limits<float>::max();
            continue;
        }

        const cv::Mat& feat = features[i]; // 1 x C
        const cv::Mat& patch = patches[i].first; // patchSize x patchSize x C
        int dim = feat.cols;

        float minDist = std::numeric_limits<float>::max();
        cv::Point2f bestOffset(0, 0);

        for (int y = 0; y < patch.rows; ++y) {
            const float* patchRowPtr = patch.ptr<float>(y);
            for (int x = 0; x < patch.cols; ++x) {
                const float* patchPtr = patchRowPtr + x * dim;
                const float* featPtr = feat.ptr<float>();

                float dist = 0.f;
                for (int c = 0; c < dim; ++c) {
                    float diff = patchPtr[c] - featPtr[c];
                    dist += diff * diff;
                }

                if (dist < minDist) {
                    minDist = dist;
                    bestOffset = cv::Point2f(x, y);
                }
            }
        }
        flow[i] = (bestOffset + patches[i].second)*scale;
        scores[i] = minDist;
    }

    return {flow, scores};
}


std::pair<std::vector<cv::Point2f>, std::vector<float> > SparseFlow::computeFlowAndScoreSubPix(const std::vector<cv::Mat> &features, const std::vector<std::pair<cv::Mat, cv::Point2f> > &patches, int patchSize, float scale) {
    int n = static_cast<int>(features.size());
        std::vector<cv::Point2f> flow(n, cv::Point2f(0, 0));
        std::vector<float> scores(n, std::numeric_limits<float>::max());
        int halfPatch = patchSize / 2;

#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            if (features[i].empty() || patches[i].first.empty()) {
                flow[i] = cv::Point2f(0, 0);
                scores[i] = std::numeric_limits<float>::max();
                continue;
            }

            const cv::Mat& feat = features[i];         // 1 x C
            const cv::Mat& patch = patches[i].first;   // patchSize x patchSize x C
            const int dim = feat.cols;

            // 构建一个 score_map（L2 距离图）
            cv::Mat score_map(patch.rows, patch.cols, CV_32F);
            for (int y = 0; y < patch.rows; ++y) {
                for (int x = 0; x < patch.cols; ++x) {
                    const float* patchPtr = patch.ptr<float>(y) + x * dim;
                    const float* featPtr = feat.ptr<float>();

                    float dist = 0.f;
                    for (int c = 0; c < dim; ++c) {
                        float diff = patchPtr[c] - featPtr[c];
                        dist += diff * diff;
                    }
                    score_map.at<float>(y, x) = dist;
                }
            }

            // 找到最小距离位置
            double minVal;
            cv::Point minLoc;
            cv::minMaxLoc(score_map, &minVal, nullptr, &minLoc, nullptr);

            cv::Point2f subpixelOffset(minLoc.x, minLoc.y);

            // 抛物线拟合 - 亚像素精度
            if (minLoc.x > 0 && minLoc.x < patch.cols - 1 &&
                minLoc.y > 0 && minLoc.y < patch.rows - 1) {

                float dx[3] = {
                    score_map.at<float>(minLoc.y, minLoc.x - 1),
                    score_map.at<float>(minLoc.y, minLoc.x),
                    score_map.at<float>(minLoc.y, minLoc.x + 1)
                };
                float dy[3] = {
                    score_map.at<float>(minLoc.y - 1, minLoc.x),
                    score_map.at<float>(minLoc.y, minLoc.x),
                    score_map.at<float>(minLoc.y + 1, minLoc.x)
                };

                auto parabola_peak = [](float v0, float v1, float v2) -> float {
                    float denom = 2 * (v0 - 2 * v1 + v2);
                    if (std::abs(denom) < 1e-6) return 0.0f;
                    return (v0 - v2) / denom;
                };

                float dx_offset = parabola_peak(dx[0], dx[1], dx[2]);
                float dy_offset = parabola_peak(dy[0], dy[1], dy[2]);

                subpixelOffset.x += dx_offset;
                subpixelOffset.y += dy_offset;
                }

            // flow = (offset + patch center offset) * scale
            flow[i] = (subpixelOffset + patches[i].second) * scale;
            scores[i] = static_cast<float>(minVal);
        }

        return {flow, scores};
}



struct Logger : public ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << msg << std::endl;
    }
} gLogger;

TRTInferV3::TRTInferV3(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) throw std::runtime_error("Engine not found");
    file.seekg(0,std::ios::end);
    size_t sz = file.tellg();
    file.seekg(0,std::ios::beg);
    std::vector<char> buf(sz);
    file.read(buf.data(), sz);

    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(buf.data(), sz);
    context = engine->createExecutionContext();
    cudaStreamCreate(&stream);
    initBindings();

    feat_buffer.reserve(inputDims.width * inputDims.height * sizeof(float) * 5);
}

TRTInferV3::~TRTInferV3() {
    for (auto ptr : buffers) cudaFree(ptr);
    cudaStreamDestroy(stream);
}

void TRTInferV3::initBindings() {
    nbIO = engine->getNbIOTensors();
    buffers.resize(nbIO);
    for (int i = 0; i < nbIO; i++) {
        const char* name = engine->getIOTensorName(i);
        auto d = engine->getTensorShape(name);
        size_t vol = 1; for (int k=0; k<d.nbDims; k++) vol*=d.d[k];
        cudaMalloc(&buffers[i], vol * sizeof(float));
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            inputIndex = i;
            inputDims = cv::Size(d.d[3], d.d[2]);
        }
    }
}

void TRTInferV3::infer(const cv::Mat& input, std::vector<float> &feat0, std::vector<float> &feat1, std::vector<float> &feat2,
    std::vector<float> &con0, std::vector<float> &con1, std::vector<float> &con2) {
    // 检查输入类型和通道数
    if (input.channels() != 1 || input.type() != CV_32FC1) {
        throw std::runtime_error("Input must be CV_32FC1 with 1 channel");
    }

    // 获取输入尺寸
    int H = input.rows;
    int W = input.cols;

    // 重塑成 [1, 1, H, W]（batch, channel, height, width）
    std::vector<float> inputTensor(1 * 1 * H * W);
    std::memcpy(inputTensor.data(), input.ptr<float>(), sizeof(float) * H * W);

    // 拷贝到 GPU
    cudaMemcpyAsync(buffers[inputIndex], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 设置绑定
    for (int i = 0; i < nbIO; ++i) {
        context->setTensorAddress(engine->getIOTensorName(i), buffers[i]);
    }

    // 推理
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("enqueueV3 failed");
    }

    cudaStreamSynchronize(stream);

    // 处理输出
    int index = 0;
    for (int i = 0; i < nbIO; ++i) {
        if (i == inputIndex) continue;
        const char* name = engine->getIOTensorName(i);
        auto d = engine->getTensorShape(name); // shape: [1, 64, H/8, W/8]

        int N = d.d[0];
        int C = d.d[1];
        int outH = d.d[2];
        int outW = d.d[3];

        size_t outSize = N * C * outH * outW;
        if (index == 0) {
            feat0.resize(outSize);
            cudaMemcpy(&feat0[0], buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);
        }
        // else if (index == 1) {
        //     feat1.reserve(outSize);
        //     cudaMemcpy(&feat1[0], buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);
        // }
        else if (index == 1) {
            feat2.resize(outSize);
            cudaMemcpy(&feat2[0], buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);
        } else if (index == 2) {
            con0.resize(outSize);
            cudaMemcpy(&con0[0], buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);
        }
        // else if (index == 4) {
        //     con1.reserve(outSize);
        //     cudaMemcpy(&con1[0], buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);
        // }
        else if (index == 3) {
            con2.resize(outSize);
            cudaMemcpy(&con2[0], buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);
        }
        index ++;
    }
}

cv::Mat TRTInferV3::infer(const cv::Mat& input) {
    // 检查输入类型和通道数
    if (input.channels() != 1 || input.type() != CV_32FC1) {
        throw std::runtime_error("Input must be CV_32FC1 with 1 channel");
    }

    // 获取输入尺寸
    int H = input.rows;
    int W = input.cols;

    // 重塑成 [1, 1, H, W]（batch, channel, height, width）
    std::vector<float> inputTensor(1 * 1 * H * W);
    std::memcpy(inputTensor.data(), input.ptr<float>(), sizeof(float) * H * W);

    // 拷贝到 GPU
    cudaMemcpyAsync(buffers[inputIndex], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 设置绑定
    for (int i = 0; i < nbIO; ++i) {
        context->setTensorAddress(engine->getIOTensorName(i), buffers[i]);
    }

    // 推理
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("enqueueV3 failed");
    }

    cudaStreamSynchronize(stream);

    // 处理输出
    for (int i = 0; i < nbIO; ++i) {
        if (i == inputIndex) continue;

        const char* name = engine->getIOTensorName(i);
        auto d = engine->getTensorShape(name); // shape: [1, 64, H/8, W/8]

        int N = d.d[0];
        int C = d.d[1];
        int outH = d.d[2];
        int outW = d.d[3];

        size_t outSize = N * C * outH * outW;

        cudaMemcpy(&feat_buffer[0], buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);

        // 转为 OpenCV 格式: H x W x C
        cv::Mat output(outH, outW, CV_32FC(C), &feat_buffer[0]);
        std::vector<cv::Mat> chw(C);
        for (int i = 0; i < C; ++i)
        {
            chw[i] = cv::Mat((int)outH, (int)outW, CV_32FC1, &feat_buffer[0] + i * outH * outW);
        }
        cv::merge(chw, output);
        // 拷贝后返回（避免指针释放）
        return output.clone();
    }

    throw std::runtime_error("No output tensor found");
}

SparsFlowTRT::SparsFlowTRT(const std::string& engine_path) {
    loadEngine(engine_path);
    allocateBuffers();
}

SparsFlowTRT::~SparsFlowTRT() {
    for (auto b : buffers_) cudaFree(b);
    cudaStreamDestroy(stream_);
}

void SparsFlowTRT::loadEngine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Engine not found");
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);

    runtime_ = createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(buf.data(), sz);
    context_ = engine_->createExecutionContext();
    cudaStreamCreate(&stream_);
}

void SparsFlowTRT::allocateBuffers() {
    int nb = engine_->getNbIOTensors();
    buffers_.resize(nb);
    dims_.resize(nb);
    is_input_.resize(nb);

    for (int i = 0; i < nb; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto d = engine_->getTensorShape(name);
        dims_[i] = d;
        size_t vol = 1;
        for (int k = 0; k < d.nbDims; ++k) vol *= d.d[k];
        cudaMalloc(&buffers_[i], vol * sizeof(float));
        is_input_[i] = engine_->getTensorIOMode(name) == TensorIOMode::kINPUT;
    }
}

cv::Mat makeInputPts(const std::vector<cv::Point2f>& pts, int total_pts = PTS_NUM) {
    // 分配 1 x total_pts x 2 的 float32 Mat
    cv::Mat input(1, total_pts, CV_32FC2, cv::Scalar(10.0f, 10.0f));  // 初始化为0

    int n = std::min((int)pts.size(), total_pts);
    // int n = PTS_NUM;
    for (int i = 0; i < n; ++i) {
        input.at<cv::Vec2f>(0, i)[0] = pts[i].x;
        input.at<cv::Vec2f>(0, i)[1] = pts[i].y;
    }

    return input;  // 注意：这是连续内存，可安全使用 ptr<float>()
}

void SparsFlowTRT::infer(const std::vector<float>& feat00, const std::vector<float>& feat10, const std::vector<float>& con1,
           const std::vector<float>& feat01, const std::vector<float>& feat11, const std::vector<float>& con2,
           const std::vector<float>& feat02, const std::vector<float>& feat12, const std::vector<float>& con3,
           const std::vector<cv::Point2f>& pts0, const std::vector<cv::Point2f>& pts1,
           std::vector<cv::Point2f>& out_pts2, std::vector<uchar>& out_scores)
{
    out_scores.clear();
    int in_idx = 0;
    cv::Mat pts0_mat = makeInputPts(pts0);
    cv::Mat pts1_mat = makeInputPts(pts1);

    std::vector<float> score(PTS_NUM);

    for (int i = 0; i < buffers_.size(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        context_->setTensorAddress(name, buffers_[i]);

        if (!is_input_[i]) continue;

        auto& d = dims_[i];
        size_t sz = 1;
        for (int k = 0; k < d.nbDims; ++k) sz *= d.d[k];

        const float* src = nullptr;
        // std::cout<<"i "<<i<<" name: "<<name<<std::endl;
        switch (in_idx++) {
            case 0:
                src = reinterpret_cast<const float*>(&feat00[0]); break;
                // case 1:
                //     src = reinterpret_cast<const float*>(&feat01[0]); break;
            case 1:
                src = reinterpret_cast<const float*>(&feat02[0]); break;
            case 2:
                src = reinterpret_cast<const float*>(&feat10[0]); break;
                // case 4:
                //     src = reinterpret_cast<const float*>(&feat11[0]); break;
            case 3:
                src = reinterpret_cast<const float*>(&feat12[0]); break;
            case 4:
                src = reinterpret_cast<const float*>(&con1[0]); break;
                // case 4:
                //     src = reinterpret_cast<const float*>(&con2[0]); break;
            case 5:
                src = reinterpret_cast<const float*>(&con3[0]); break;
            case 6:
                src = reinterpret_cast<const float*>(pts0_mat.ptr<float>()); break;
            case 7:
                src = reinterpret_cast<const float*>(pts1_mat.ptr<float>()); break;
            default:
                throw std::runtime_error("Too many input tensors.");
        }

        cudaMemcpyAsync(buffers_[i], src, sz * sizeof(float), cudaMemcpyHostToDevice, stream_);
    }

    if (!context_->enqueueV3(stream_))
        throw std::runtime_error("Inference failed!");

    cudaStreamSynchronize(stream_);

    // 处理输出
    int out_idx = 0;
    for (int i = 0; i < buffers_.size(); ++i) {
        if (is_input_[i]) continue;

        const auto& d = dims_[i];
        size_t sz = 1;
        for (int k = 0; k < d.nbDims; ++k) sz *= d.d[k];

        std::vector<float> host_output(sz);
        cudaMemcpy(host_output.data(), buffers_[i], sz * sizeof(float), cudaMemcpyDeviceToHost);

        if (out_idx == 0) {
            // 第一个输出是 [1, PTS_NUM, 3]
            out_pts2.resize(d.d[1]);

            for (int j = 0; j < d.d[1]; ++j) {
                out_pts2[j] = cv::Point2f(host_output[j * 2], host_output[j * 2 + 1]);
                // out_scores[j] = (host_output[j * 3 + 2] > 0.6);
                // std::cout<<" "<<host_output[j * 2]<<" "<<host_output[j * 2 + 1]<<std::endl;
            }
        } else if (out_idx == 1) {
            // 第二个输出是 [1, PTS_NUM]
            // score = std::move(host_output);
            for (int j =0; j < PTS_NUM;j ++) {
                out_scores.push_back(host_output[j]*2 > 0.7 ? 1 : 0);
                // std::cout<<host_output[j]<<std::endl;
            }
        }
        ++out_idx;
    }
}

// void SparsFlowTRT::infer(const cv::Mat& img0, const cv::Mat& img1,
//                          const std::vector<float>& feat0, const std::vector<float>& feat1,
//                          const std::vector<cv::Point2f>& pts0, const std::vector<cv::Point2f>& pts1,
//                          std::vector<cv::Point2f>& out_pts2, std::vector<uchar>& out_scores) {
//     out_scores.clear();
//     int in_idx = 0;
//     cv::Mat pts0_mat = makeInputPts(pts0);
//     cv::Mat pts1_mat = makeInputPts(pts1);
//
//     int H = img0.rows;
//     int W = img0.cols;
//
//     std::vector<float> input_img0(1 * 1 * H * W);
//     std::memcpy(input_img0.data(), img0.ptr<float>(), sizeof(float) * H * W);
//
//     std::vector<float> input_img1(1 * 1 * H * W);
//     std::memcpy(input_img1.data(), img1.ptr<float>(), sizeof(float) * H * W);
//
//     for (int i = 0; i < buffers_.size(); ++i) {
//         const char* name = engine_->getIOTensorName(i);
//         context_->setTensorAddress(name, buffers_[i]);
//
//         if (!is_input_[i]) continue;
//
//         auto& d = dims_[i];
//         size_t sz = 1;
//         for (int k = 0; k < d.nbDims; ++k) sz *= d.d[k];
//
//         const float* src = nullptr;
//         switch (in_idx++) {
//             case 0: src = reinterpret_cast<const float*>(img1.ptr<float>()); break;
//             case 1: src = feat0.data(); break;
//             case 2: src = feat1.data(); break;
//             case 3: src = reinterpret_cast<const float*>(pts0_mat.ptr<float>()); break;
//             case 4: src = reinterpret_cast<const float*>(pts1_mat.ptr<float>()); break;
//             default: throw std::runtime_error("Too many input tensors.");
//         }
//
//         cudaMemcpyAsync(buffers_[i], src, sz * sizeof(float), cudaMemcpyHostToDevice, stream_);
//     }
//
//     if (!context_->enqueueV3(stream_))
//         throw std::runtime_error("Inference failed!");
//
//     cudaStreamSynchronize(stream_);
//
//     // 分别存储两个输出
//     std::vector<float> host_output_pts;
//     std::vector<float> host_output_scores;
//
//     int out_idx = 0;
//     for (int i = 0; i < buffers_.size(); ++i) {
//         if (is_input_[i]) continue;
//
//         const auto& d = dims_[i];
//         size_t sz = 1;
//         for (int k = 0; k < d.nbDims; ++k) sz *= d.d[k];
//
//         std::vector<float> host_output(sz);
//         cudaMemcpy(host_output.data(), buffers_[i], sz * sizeof(float), cudaMemcpyDeviceToHost);
//
//         if (out_idx == 0) {
//             host_output_pts = host_output;
//             int N = d.d[1];  // num points
//             out_pts2.resize(N);
//             for (int j = 0; j < N; ++j) {
//                 out_pts2[j] = cv::Point2f(host_output[j * 2], host_output[j * 2 + 1]);
//             }
//         } else if (out_idx == 1) {
//             host_output_scores = host_output;
//         }
//
//         ++out_idx;
//     }
//
//     // 后处理得分
//     out_scores.resize(host_output_scores.size());
//     for (size_t j = 0; j < host_output_scores.size(); ++j) {
//         out_scores[j] = host_output_scores[j] > 0.6f ? 1 : 0;
//     }
//
//     if (!out_scores.empty()) {
//         std::cout << "scores: " << static_cast<int>(out_scores[0]) << " "
//                   << static_cast<float>(host_output_scores[0]) << std::endl;
//     }
//     std::cout << "..." << std::endl;
// }


LKFlowInfer::LKFlowInfer(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) throw std::runtime_error("Engine not found");
    file.seekg(0,std::ios::end);
    size_t sz = file.tellg();
    file.seekg(0,std::ios::beg);
    std::vector<char> buf(sz);
    file.read(buf.data(), sz);

    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(buf.data(), sz);
    context = engine->createExecutionContext();
    cudaStreamCreate(&stream);
    initBindings();

    feat_buffer.reserve(inputDims.width * inputDims.height * sizeof(float) * 5);
}

LKFlowInfer::~LKFlowInfer() {
    for (const auto ptr : buffers) cudaFree(ptr);
    cudaStreamDestroy(stream);
}

void LKFlowInfer::initBindings() {
    nbIO = engine->getNbIOTensors();
    buffers.resize(nbIO);
    for (int i = 0; i < nbIO; i++) {
        const char* name = engine->getIOTensorName(i);
        auto d = engine->getTensorShape(name);
        size_t vol = 1; for (int k=0; k<d.nbDims; k++) vol*=d.d[k];
        cudaMalloc(&buffers[i], vol * sizeof(float));
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            inputIndex = i;
            inputDims = cv::Size(d.d[3], d.d[2]);
        }
    }
}

void LKFlowInfer::infer(const cv::Mat &input, std::vector<cv::Mat> &feats) {
    feats.clear();
    feats.reserve(4);
    // 检查输入类型和通道数
    if (input.channels() != 1 || input.type() != CV_32FC1) {
        throw std::runtime_error("Input must be CV_32FC1 with 1 channel");
    }

    // 获取输入尺寸
    int H = input.rows;
    int W = input.cols;

    // 重塑成 [1, 1, H, W]（batch, channel, height, width）
    std::vector<float> inputTensor(1 * 1 * H * W);
    std::memcpy(inputTensor.data(), input.ptr<float>(), sizeof(float) * H * W);

    // 拷贝到 GPU
    cudaMemcpyAsync(buffers[inputIndex], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 设置绑定
    for (int i = 0; i < nbIO; ++i) {
        context->setTensorAddress(engine->getIOTensorName(i), buffers[i]);
    }

    // 推理
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("enqueueV3 failed");
    }

    cudaStreamSynchronize(stream);

    // 处理输出
    for (int i = 0; i < nbIO; ++i) {
        if (i == inputIndex) continue;

        const char* name = engine->getIOTensorName(i);
        auto d = engine->getTensorShape(name); // shape: [1, 64, H/8, W/8]

        int N = d.d[0];
        int C = d.d[3];
        int outH = d.d[1];
        int outW = d.d[2];

        size_t outSize = N * C * outH * outW;

        cudaMemcpy(&feat_buffer[0], buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);

        // 转为 OpenCV 格式: H x W x C
        cv::Mat output(outH, outW, CV_32FC(C), &feat_buffer[0]);
        cv::Mat mat_int;
        output.convertTo(mat_int, CV_8UC4);
        feats.emplace_back(mat_int);
    }
}

int LKFlowInfer::build_pyr(
    cv::InputArrayOfArrays _pyramidImages,
    cv::OutputArrayOfArrays pyramid,
    cv::Size winSize,
    int maxLevel,
    bool withDerivatives,
    int pyrBorder,
    int derivBorder)
{
    CV_Assert(winSize.width > 2 && winSize.height > 2);

    // 获取输入金字塔
    std::vector<cv::Mat> pyramidImages;
    _pyramidImages.getMatVector(pyramidImages);
    CV_Assert(!pyramidImages.empty() && pyramidImages[0].depth() == CV_8U);

    // 确保金字塔层级不超过输入
    maxLevel = std::min(maxLevel, static_cast<int>(pyramidImages.size()) - 1);
    int pyrstep = withDerivatives ? 2 : 1;

    // 创建输出金字塔（预分配空间）
    pyramid.create(1, (maxLevel + 1) * pyrstep, 0, -1, true);

    // 修复1: 直接使用OpenCV原生类型代替内部类型
    int derivType = CV_16SC2; // 默认使用16位有符号双通道
    if (pyramidImages[0].channels() > 1) {
        // 多通道使用8位有符号，通道数*2 (x和y方向)
        derivType = CV_MAKETYPE(CV_16S, pyramidImages[0].channels() * 2);
    }

    // 处理每一层金字塔
    for (int level = 0; level <= maxLevel; ++level)
    {
        const cv::Mat& srcImg = pyramidImages[level];
        cv::Size sz = srcImg.size();

        // 检查是否小于窗口尺寸（终止条件）
        if (sz.width < winSize.width || sz.height < winSize.height) {
            pyramid.create(1, level * pyrstep, 0, -1, true);
            return level - 1;
        }

        // 处理图像层 ---------------------------------------------------
        cv::Mat& dstImg = pyramid.getMatRef(level * pyrstep);

        // 创建带边界的图像
        dstImg.create(sz.height + winSize.height * 2,
                      sz.width + winSize.width * 2,
                      srcImg.type());

        // 设置ROI（中心区域）
        cv::Mat imgROI = dstImg(cv::Rect(winSize.width, winSize.height, sz.width, sz.height));

        // 复制并扩展边界
        if (pyrBorder == cv::BORDER_TRANSPARENT) {
            srcImg.copyTo(imgROI);
        } else {
            cv::copyMakeBorder(srcImg, dstImg,
                              winSize.height, winSize.height,
                              winSize.width, winSize.width,
                              pyrBorder | cv::BORDER_ISOLATED);
        }
        dstImg.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);

        // 处理梯度层 ---------------------------------------------------
        if (withDerivatives)
        {
            CV_Assert(level * pyrstep + 1 < pyramid.total());
            cv::Mat& deriv = pyramid.getMatRef(level * pyrstep + 1);

            // 创建带边界的梯度图像
            deriv.create(sz.height + winSize.height * 2,
                         sz.width + winSize.width * 2,
                         derivType);

            // 设置ROI（中心区域）
            cv::Mat derivROI = deriv(cv::Rect(winSize.width, winSize.height, sz.width, sz.height));

            // 计算梯度
            if (srcImg.channels() == 1) {
                // 修复2: 直接使用OpenCV原生Scharr计算
                cv::Mat dx, dy;
                cv::Scharr(imgROI, dx, CV_16S, 1, 0);
                cv::Scharr(imgROI, dy, CV_16S, 0, 1);

                // 合并xy梯度
                std::vector<cv::Mat> xy = {dx, dy};
                cv::merge(xy, derivROI);
            } else {
                // 多通道梯度计算
                std::vector<cv::Mat> channels;
                cv::split(imgROI, channels);

                std::vector<cv::Mat> derivatives;
                for (const cv::Mat& ch : channels) {
                    cv::Mat dx, dy;
                    cv::Scharr(ch, dx, CV_16S, 1, 0);
                    cv::Scharr(ch, dy, CV_16S, 0, 1);
                    derivatives.push_back(dx);
                    derivatives.push_back(dy);
                }
                cv::merge(derivatives, derivROI);
            }

            // 扩展梯度图边界
            if (derivBorder != cv::BORDER_TRANSPARENT) {
                cv::copyMakeBorder(derivROI, deriv,
                                  winSize.height, winSize.height,
                                  winSize.width, winSize.width,
                                  derivBorder | cv::BORDER_ISOLATED);
            }
            deriv.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
        }
    }
    return maxLevel;
}



LETFlowInfer::LETFlowInfer(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) throw std::runtime_error("Engine not found");
    file.seekg(0, std::ios::end);
    size_t sz = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    file.read(buf.data(), sz);

    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(buf.data(), sz);
    context = engine->createExecutionContext();
    cudaStreamCreate(&stream);
    initBindings();

    // Reserve buffer for output
    output_buffer.reserve(inputDims.width * inputDims.height * 4);
    output_buffer2.reserve(inputDims.width * inputDims.height * 2);
}

LETFlowInfer::~LETFlowInfer() {
    for (auto ptr : buffers) cudaFree(ptr);
    cudaStreamDestroy(stream);
}

void LETFlowInfer::initBindings() {
    nbIO = engine->getNbIOTensors();
    // if (nbIO != 2) throw std::runtime_error("Unexpected number of I/O tensors (expected 1 input + 1 output)");
    buffers.resize(nbIO);
    for (int i = 0; i < nbIO; i++) {
        const char* name = engine->getIOTensorName(i);
        auto d = engine->getTensorShape(name);
        size_t vol = 1; for (int k = 0; k < d.nbDims; k++) vol *= d.d[k];
        cudaMalloc(&buffers[i], vol * sizeof(float));
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            inputIndex = i;
            inputDims = cv::Size(d.d[2], d.d[1]);  // W, H for NHWC
        }
    }
}


cv::Mat LETFlowInfer::infer(const cv::Mat& input) {
    // Check input type and channels
    if (input.channels() != 3 || input.type() != CV_32FC3) {
        throw std::runtime_error("Input must be CV_32FC3 with 3 channels");
    }

    // Get input dimensions
    int H = input.rows;
    int W = input.cols;
    if (H != inputDims.height || W != inputDims.width) {
        throw std::runtime_error("Input size mismatch");
    }

    // Prepare input tensor in NHWC format (1, H, W, 3)
    std::vector<float> inputTensor(H * W * 3);
    std::memcpy(inputTensor.data(), input.ptr<float>(), H * W * 3 * sizeof(float));

    // Copy to GPU
    cudaMemcpyAsync(buffers[inputIndex], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Set tensor addresses
    for (int i = 0; i < nbIO; ++i) {
        context->setTensorAddress(engine->getIOTensorName(i), buffers[i]);
    }

    // Run inference
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("enqueueV3 failed");
    }

    cudaStreamSynchronize(stream);

    // Process output
    for (int i = 0; i < nbIO; ++i) {
        if (i == inputIndex) continue;

        const char* name = engine->getIOTensorName(i);
        auto d = engine->getTensorShape(name);  // Expected: [1, H, W, 4]

        int N = d.d[0];
        int outH = d.d[1];
        int outW = d.d[2];
        int C = d.d[3];

        if (N != 1 || outH != H || outW != W || C != 3) {
            throw std::runtime_error("Unexpected output shape");
        }

        size_t outSize = outH * outW * C;
        output_buffer.resize(outSize);
        cudaMemcpy(output_buffer.data(), buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);

        // Output is already in NHWC layout, so directly create cv::Mat
        cv::Mat output(outH, outW, CV_32FC3, output_buffer.data());

        // Return a clone to avoid buffer issues
        return output.clone();
    }

    throw std::runtime_error("No output tensor found");
}

void LETFlowInfer::infer2(const cv::Mat& input, cv::Mat& cov, cv::Mat& desc) {
        // Check input type and channels
    if (input.channels() != 3 || input.type() != CV_32FC3) {
        throw std::runtime_error("Input must be CV_32FC3 with 3 channels");
    }

    // Get input dimensions
    int H = input.rows;
    int W = input.cols;
    if (H != inputDims.height || W != inputDims.width) {
        throw std::runtime_error("Input size mismatch");
    }

    // Prepare input tensor in NHWC format (1, H, W, 3)
    std::vector<float> inputTensor(H * W * 3);
    std::memcpy(inputTensor.data(), input.ptr<float>(), H * W * 3 * sizeof(float));

    // Copy to GPU
    cudaMemcpyAsync(buffers[inputIndex], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Set tensor addresses
    for (int i = 0; i < nbIO; ++i) {
        context->setTensorAddress(engine->getIOTensorName(i), buffers[i]);
    }

    // Run inference
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("enqueueV3 failed");
    }

    cudaStreamSynchronize(stream);
    
    // Process output
    for (int i = 0; i < nbIO; ++i) {
        if (i == inputIndex) continue;

        const char* name = engine->getIOTensorName(i);
        auto d = engine->getTensorShape(name);  // Expected: [1, H, W, 4]

        int N = d.d[0];
        int outH = d.d[1];
        int outW = d.d[2];
        int C = d.d[3];

        if (N != 1 || outH != H || outW != W) {
            throw std::runtime_error("Unexpected output shape");
        }
        
        if (C == 3) {
            size_t outSize = outH * outW * C;
            output_buffer.resize(outSize);
            cudaMemcpy(output_buffer.data(), buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);

            // Output is already in NHWC layout, so directly create cv::Mat
            cv::Mat output(outH, outW, CV_32FC3, output_buffer.data());
            desc = output.clone();
        }

        if (C == 1) {
            size_t outSize = outH * outW * C;
            output_buffer2.resize(outSize);
            cudaMemcpy(output_buffer2.data(), buffers[i], outSize * sizeof(float), cudaMemcpyDeviceToHost);

            // Output is already in NHWC layout, so directly create cv::Mat
            cv::Mat output(outH, outW, CV_32FC1, output_buffer2.data());
            cov = output.clone();
        }
    }
}