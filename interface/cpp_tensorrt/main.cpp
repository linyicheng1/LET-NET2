#include "tracking_trt.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: ./demo_trt <engine_path> <video_path>" << std::endl;
        std::cout << "Example: ./demo_trt letnet_480x640_fp16.engine video.mp4" << std::endl;
        return -1;
    }
    
    std::string engine_path = argv[1];
    std:: string video_path = argv[2];
    
    // Initialize tracker
    corner_tracking_trt tracker;
    
    // Load TensorRT engine
    std::cout << "Loading TensorRT engine..." << std::endl;
    if (!tracker.load_model(engine_path)) {
        std::cerr << "Failed to load engine!" << std::endl;
        return -1;
    }
    
    // Open video
    std::cout << "Opening video:  " << video_path << std:: endl;
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        int cam_idx = std::atoi(video_path.c_str());
        cap.open(cam_idx);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video!" << std:: endl;
            return -1;
        }
    }
    
    int fps = (int)cap.get(cv::CAP_PROP_FPS);
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    std::cout << "Video:  " << width << "x" << height << " @ " << fps << "fps" << std::endl;
    
    // Create video writer
    std::string output_path = "output_trt.avi";
    cv::VideoWriter writer(output_path, cv:: VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          fps > 0 ? fps : 30, cv::Size(width, height));
    
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer!" << std::endl;
        return -1;
    }
    
    std::cout << "Output:  " << output_path << std:: endl;
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
        
        cv::Mat gray;
        cv:: cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        auto t1 = std::chrono::high_resolution_clock::now();
        cv::Mat desc = tracker.get_descriptor(frame);
        cv::imwrite("desc.jpg", desc);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        if (desc.empty()) {
            std::cerr << "Failed to get descriptor for frame " << frame_count << std::endl;
            continue;
        }
        
        tracker.update(gray, desc);
        auto t3 = std:: chrono::high_resolution_clock::now();
        
        cv::Mat vis = tracker.draw_tracking(frame);
        writer.write(vis);
        
        auto t_end = std::chrono::high_resolution_clock::now();
        
        double time_inference = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double time_tracking = std:: chrono::duration<double, std::milli>(t3 - t2).count();
        double time_total = std::chrono:: duration<double, std::milli>(t_end - t_start).count();
        
        total_inference_time += time_inference;
        total_tracking_time += time_tracking;
        
        if (frame_count % 30 == 0) {
            std::cout << "Frame " << frame_count
                     << " | Points: " << tracker.get_num_points()
                     << " | Inference: " << time_inference << "ms"
                     << " | Tracking: " << time_tracking << "ms"
                     << " | FPS: " << (1000.0 / time_total) << std::endl;
        }
        
        int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
        
        frame_count++;
    }
    
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    
    std::cout << "\n=================" << std::endl;
    std::cout << "Processing Complete!" << std::endl;
    std::cout << "Total frames:  " << frame_count << std:: endl;
    std::cout << "Avg inference: " << (total_inference_time / frame_count) << "ms" << std::endl;
    std::cout << "Avg tracking: " << (total_tracking_time / frame_count) << "ms" << std::endl;
    std::cout << "Avg FPS: " << (frame_count * 1000.0 / (total_inference_time + total_tracking_time)) << std::endl;
    std::cout << "Output: " << output_path << std::endl;
    std::cout << "=================" << std::endl;
    
    return 0;
}
