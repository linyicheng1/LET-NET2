#include "tracking.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: ./demo <model. param> <model.bin> <video_path>" << std::endl;
        std::cout << "Example: ./demo letnet_480x640.param letnet_480x640.bin video.mp4" << std::endl;
        return -1;
    }
    
    std::string param_path = argv[1];
    std:: string bin_path = argv[2];
    std::string video_path = argv[3];
    
    // 初始化跟踪器
    corner_tracking tracker;
    
    // 加载ncnn模型
    std::cout << "Loading model..." << std::endl;
    if (! tracker.load_model(param_path, bin_path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }
    
    // 打开视频
    std::cout << "Opening video:  " << video_path << std:: endl;
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        // 尝试作为摄像头索引
        int cam_idx = std::atoi(video_path.c_str());
        cap.open(cam_idx);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video:  " << video_path << std:: endl;
            return -1;
        }
    }
    
    // 获取视频属性
    int fps = (int)cap.get(cv::CAP_PROP_FPS);
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    std::cout << "Video properties: " << width << "x" << height << " @ " << fps << "fps" << std::endl;
    
    // 创建视频输出
    std::string output_path = "output.avi";
    cv::VideoWriter writer(output_path, 
                          cv::VideoWriter:: fourcc('M', 'J', 'P', 'G'), 
                          fps > 0 ? fps : 30, 
                          cv::Size(width, height));
    
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer!" << std::endl;
        return -1;
    }
    
    std::cout << "Output will be saved to: " << output_path << std::endl;
    std::cout << "Processing..." << std::endl;
    
    cv::Mat frame;
    int frame_count = 0;
    double total_inference_time = 0.0;
    double total_tracking_time = 0.0;
    
    while (true) {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        
        // 转换为灰度图用于GFTT
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // 获取descriptor
        auto t1 = std::chrono::high_resolution_clock::now();
        cv::Mat desc = tracker.get_descriptor(frame);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        if (desc.empty()) {
            std::cerr << "Failed to get descriptor for frame " << frame_count << std::endl;
            continue;
        }
        cv::imwrite("desc.jpg", desc);
        
        // 更新跟踪
        tracker.update(gray, desc);
        auto t3 = std::chrono::high_resolution_clock::now();
        
        // 绘制跟踪结果
        cv::Mat vis = tracker.draw_tracking(frame);
        
        // 保存到视频
        writer.write(vis);
        
        auto t_end = std::chrono::high_resolution_clock::now();
        
        // 计算耗时
        double time_inference = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double time_tracking = std:: chrono::duration<double, std::milli>(t3 - t2).count();
        double time_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        
        total_inference_time += time_inference;
        total_tracking_time += time_tracking;
        
        // 每30帧打印一次信息
        if (frame_count % 30 == 0) {
            std::cout << "Frame " << frame_count 
                     << " | Points: " << tracker.get_num_points()
                     << " | Inference: " << time_inference << "ms"
                     << " | Tracking: " << time_tracking << "ms"
                     << " | Total: " << time_total << "ms"
                     << " | FPS: " << (1000.0 / time_total) << std::endl;
        }
        
        // 可选：显示预览窗口
        // cv::imshow("Tracking Preview", vis);
        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            std::cout << "ESC pressed, stopping..." << std::endl;
            break;
        }
        
        frame_count++;
    }
    
    // 清理
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    
    // 打印统计信息
    std::cout << "\n" << "=================" << std::endl;
    std::cout << "Processing Complete!" << std::endl;
    // std::cout << "="*60 << std::endl;
    // std::cout << "Total frames processed: " << frame_count << std::endl;
    // std:: cout << "Average inference time: " << (total_inference_time / frame_count) << "ms" << std::endl;
    // std::cout << "Average tracking time: " << (total_tracking_time / frame_count) << "ms" << std::endl;
    // std:: cout << "Average FPS: " << (frame_count * 1000.0 / (total_inference_time + total_tracking_time)) << std::endl;
    std::cout << "Output saved to: " << output_path << std::endl;
    std::cout << "=================" << std::endl;
    
    return 0;
}