// /*******************************************************
//  * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
//  *
//  * This file is part of VINS.
//  *
//  * Licensed under the GNU General Public License v3.0;
//  * you may not use this file except in compliance with the License.
//  *
//  * Author: Qin Tong (qintonguav@gmail.com)
//  *******************************************************/

// #include <stdio.h>
// #include <queue>
// #include <map>
// #include <thread>
// #include <mutex>
// #include <ros/ros.h>
// #include <cv_bridge/cv_bridge.h>
// #include <opencv2/opencv.hpp>
// #include "estimator/estimator.h"
// #include "estimator/parameters.h"
// #include "utility/visualization.h"

// Estimator estimator;

// queue<sensor_msgs::ImuConstPtr> imu_buf;
// queue<sensor_msgs::PointCloudConstPtr> feature_buf;
// queue<sensor_msgs::ImageConstPtr> img0_buf;
// queue<sensor_msgs::ImageConstPtr> img1_buf;
// std::mutex m_buf;


// void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
// {
//     m_buf.lock();
//     img0_buf.push(img_msg);
//     m_buf.unlock();
// }

// void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
// {
//     m_buf.lock();
//     img1_buf.push(img_msg);
//     m_buf.unlock();
// }


// cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
// {
//     cv_bridge::CvImageConstPtr ptr;
//     if (img_msg->encoding == "8UC1")
//     {
//         sensor_msgs::Image img;
//         img.header = img_msg->header;
//         img.height = img_msg->height;
//         img.width = img_msg->width;
//         img.is_bigendian = img_msg->is_bigendian;
//         img.step = img_msg->step;
//         img.data = img_msg->data;
//         img.encoding = "mono8";
//         ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
//     }
//     else
//         ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

//     cv::Mat img = ptr->image.clone();
//     return img;
// }

// // extract images with same timestamp from two topics
// void sync_process()
// {
//     while(1)
//     {
//         if(STEREO)
//         {
//             cv::Mat image0, image1;
//             std_msgs::Header header;
//             double time = 0;
//             m_buf.lock();
//             if (!img0_buf.empty() && !img1_buf.empty())
//             {
//                 double time0 = img0_buf.front()->header.stamp.toSec();
//                 double time1 = img1_buf.front()->header.stamp.toSec();
//                 // 0.003s sync tolerance
//                 if(time0 < time1 - 0.003)
//                 {
//                     img0_buf.pop();
//                     printf("throw img0\n");
//                 }
//                 else if(time0 > time1 + 0.003)
//                 {
//                     img1_buf.pop();
//                     printf("throw img1\n");
//                 }
//                 else
//                 {
//                     time = img0_buf.front()->header.stamp.toSec();
//                     header = img0_buf.front()->header;
//                     image0 = getImageFromMsg(img0_buf.front());
//                     img0_buf.pop();
//                     image1 = getImageFromMsg(img1_buf.front());
//                     img1_buf.pop();
//                     //printf("find img0 and img1\n");
//                 }
//             }
//             m_buf.unlock();
//             if(!image0.empty())
//                 estimator.inputImage(time, image0, image1);
//         }
//         else
//         {
//             cv::Mat image;
//             std_msgs::Header header;
//             double time = 0;
//             m_buf.lock();
//             if(!img0_buf.empty())
//             {
//                 time = img0_buf.front()->header.stamp.toSec();
//                 header = img0_buf.front()->header;
//                 image = getImageFromMsg(img0_buf.front());
//                 img0_buf.pop();
//             }
//             m_buf.unlock();
//             if(!image.empty())
//                 estimator.inputImage(time, image);
//         }

//         std::chrono::milliseconds dura(2);
//         std::this_thread::sleep_for(dura);
//     }
// }


// void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
// {
//     double t = imu_msg->header.stamp.toSec();
//     double dx = imu_msg->linear_acceleration.x;
//     double dy = imu_msg->linear_acceleration.y;
//     double dz = imu_msg->linear_acceleration.z;
//     double rx = imu_msg->angular_velocity.x;
//     double ry = imu_msg->angular_velocity.y;
//     double rz = imu_msg->angular_velocity.z;
//     Vector3d acc(dx, dy, dz);
//     Vector3d gyr(rx, ry, rz);
//     estimator.inputIMU(t, acc, gyr);
//     return;
// }


// void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
// {
//     map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
//     for (unsigned int i = 0; i < feature_msg->points.size(); i++)
//     {
//         int feature_id = feature_msg->channels[0].values[i];
//         int camera_id = feature_msg->channels[1].values[i];
//         double x = feature_msg->points[i].x;
//         double y = feature_msg->points[i].y;
//         double z = feature_msg->points[i].z;
//         double p_u = feature_msg->channels[2].values[i];
//         double p_v = feature_msg->channels[3].values[i];
//         double velocity_x = feature_msg->channels[4].values[i];
//         double velocity_y = feature_msg->channels[5].values[i];
//         if(feature_msg->channels.size() > 5)
//         {
//             double gx = feature_msg->channels[6].values[i];
//             double gy = feature_msg->channels[7].values[i];
//             double gz = feature_msg->channels[8].values[i];
//             pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
//             //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
//         }
//         ROS_ASSERT(z == 1);
//         Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
//         xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
//         featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
//     }
//     double t = feature_msg->header.stamp.toSec();
//     estimator.inputFeature(t, featureFrame);
//     return;
// }

// void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
// {
//     if (restart_msg->data == true)
//     {
//         ROS_WARN("restart the estimator!");
//         estimator.clearState();
//         estimator.setParameter();
//     }
//     return;
// }

// void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
// {
//     if (switch_msg->data == true)
//     {
//         //ROS_WARN("use IMU!");
//         estimator.changeSensorType(1, STEREO);
//     }
//     else
//     {
//         //ROS_WARN("disable IMU!");
//         estimator.changeSensorType(0, STEREO);
//     }
//     return;
// }

// void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
// {
//     if (switch_msg->data == true)
//     {
//         //ROS_WARN("use stereo!");
//         estimator.changeSensorType(USE_IMU, 1);
//     }
//     else
//     {
//         //ROS_WARN("use mono camera (left)!");
//         estimator.changeSensorType(USE_IMU, 0);
//     }
//     return;
// }



// //-cam0
// //   |-data
// //   |-data.csv
// //-cam1
// //   |-data
// //   |-data.csv
// // data.csv 的参考示例为
// // #timestamp [ns]	filename	exposure [ns]
// // 1.54904E+18	1549037852844647445.png	4030000
// // 1.54904E+18	1549037852924647445.png	3405000
// // 1.54904E+18	1549037853004647445.png	2905000
// // 1.54904E+18	1549037853084647445.png	2467000
// // 1.54904E+18	1549037853164647445.png	2155000
// // 1.54904E+18	1549037853244647445.png	1905000
// // 1.54904E+18	1549037853324647445.png	1655000


// class UMAGrabber {
// public:
//     UMAGrabber(std::string path) {
//         std::cout << "\nUMA Grabber initialized\n";
//         dataset_path_ = path;
//     }

//     void sync_process() {
//         std::cout << "\nStarting dataset processing...\n";
//         std::string root_path = dataset_path_;

//         load_images(
//             root_path + "/cam0/data.csv",
//             root_path + "/cam1/data.csv",
//             root_path + "/cam0/data",
//             root_path + "/cam1/data",
//             root_path + "/imu0/data.csv",
//             image_left_,
//             image_right_,
//             image_timestamps_,
//             imu_time_,
//             imu_data_
//         );

//         // Wait for system initialization
//         std::this_thread::sleep_for(std::chrono::seconds(5));

//         ros::Time start = ros::Time::now();
//         size_t image_index = 1;
//         size_t imu_index = 0;

//         sensor_msgs::Image img_msg_left;
//         sensor_msgs::Image img_msg_right;

//         img_msg_left.header.stamp.sec  = static_cast<uint32_t>(image_timestamps_[0] / 1e9);
//         img_msg_left.header.stamp.nsec = static_cast<uint32_t>(fmod(image_timestamps_[0], 1e9));
//         img_msg_left.header.frame_id = "world";
//         img_msg_left.height = ROW;
//         img_msg_left.width = COL;
//         img_msg_left.encoding = "mono8";
//         img_msg_left.is_bigendian = false;
//         img_msg_left.step = COL;
//         img_msg_left.data.resize(ROW * COL);

//         img_msg_right.header.stamp.sec  = static_cast<uint32_t>(image_timestamps_[0] / 1e9);
//         img_msg_right.header.stamp.nsec = static_cast<uint32_t>(fmod(image_timestamps_[0], 1e9));
//         img_msg_right.header.frame_id = "world";
//         img_msg_right.height = ROW;
//         img_msg_right.width = COL;
//         img_msg_right.encoding = "mono8";
//         img_msg_right.is_bigendian = false;
//         img_msg_right.step = COL;
//         img_msg_right.data.resize(ROW * COL);

//         // first image
//         memcpy(&img_msg_left.data[0], image_left_buffer_[0].data, ROW * COL);
//         sensor_msgs::Image::ConstPtr img_ptr = boost::make_shared<sensor_msgs::Image>(img_msg_left);
//         {
//             m_buf.lock();
//             img0_buf.push(img_ptr);
//             m_buf.unlock();
//         }

//         memcpy(&img_msg_right.data[0], image_right_buffer_[0].data, ROW * COL);
//         img_ptr = boost::make_shared<sensor_msgs::Image>(img_msg_right);
//         {
//             m_buf.lock();
//             img1_buf.push(img_ptr);
//             m_buf.unlock();
//         }

//         while (image_index < image_timestamps_.size()) {
//             ros::Time now = ros::Time::now();
//             ros::Duration d = now - start;

//             long double d_ns = d.sec * 1e9 + d.nsec;
//             // if (USE_IMU)
//             {
//                 if (d_ns > imu_time_[imu_index] - image_timestamps_[0]) {
//                     double dx = imu_data_[imu_index].linear_acceleration.x;
//                     double dy = imu_data_[imu_index].linear_acceleration.y;
//                     double dz = imu_data_[imu_index].linear_acceleration.z;
//                     double rx = imu_data_[imu_index].angular_velocity.x;
//                     double ry = imu_data_[imu_index].angular_velocity.y;
//                     double rz = imu_data_[imu_index].angular_velocity.z;
//                     Vector3d acc(dx, dy, dz);
//                     Vector3d gyr(rx, ry, rz);
//                     // ros::Duration imu_time;
//                     // auto stamp = start + imu_time.fromNSec((int64)imu_time_[imu_index]);
//                     // double t = stamp.toSec();
//                     long double t = imu_time_[imu_index] / 1e9;
//                     estimator.inputIMU((double)t, acc, gyr);
//                     imu_index ++;
//                 }
//             }

//             if (d_ns < image_timestamps_[image_index] - image_timestamps_[0]) {
//                 continue;
//             }

//             // if (STEREO)
//             {
//                 std::lock_guard<std::mutex> lock(img_mutex);
//                 // cv::Mat img_left = cv::imread(image_left_[image_index], cv::IMREAD_GRAYSCALE);
//                 // cv::Mat img_right = cv::imread(image_right_[image_index], cv::IMREAD_GRAYSCALE);
//                 cv::Mat img_left = image_left_buffer_[image_index];
//                 cv::Mat img_right = image_right_buffer_[image_index];

//                 if (!img_left.empty() && !img_right.empty()) {
//                     ros::Duration img_time;
//                     ///// left
//                     img_msg_left.header.stamp.sec  = static_cast<uint32_t>(image_timestamps_[image_index] / 1e9);
//                     img_msg_left.header.stamp.nsec = static_cast<uint32_t>(fmod(image_timestamps_[image_index], 1e9));
//                     memcpy(&img_msg_left.data[0], image_left_buffer_[image_index].data, ROW * COL);
//                     img_ptr = boost::make_shared<sensor_msgs::Image>(img_msg_left);
//                     {
//                         m_buf.lock();
//                         img0_buf.push(img_ptr);
//                         m_buf.unlock();
//                     }
//                     ////// right
//                     img_msg_right.header.stamp.sec  = static_cast<uint32_t>(image_timestamps_[image_index] / 1e9);
//                     img_msg_right.header.stamp.nsec = static_cast<uint32_t>(fmod(image_timestamps_[image_index], 1e9));
//                     memcpy(&img_msg_right.data[0], image_right_buffer_[image_index].data, ROW * COL);
//                     img_ptr = boost::make_shared<sensor_msgs::Image>(img_msg_right);
//                     {
//                         m_buf.lock();
//                         img1_buf.push(img_ptr);
//                         m_buf.unlock();
//                     }
//                 } else {
//                     std::cerr << "Empty stereo images at index: " << image_index << std::endl;
//                 }
//             }
//             // else {
//             //     std::lock_guard<std::mutex> lock(img_mutex);
//             //     cv::Mat img_left = image_left_buffer_[image_index];
//             //
//             //     if (!img_left.empty()) {
//             //         estimator.inputImage(image_timestamps_[image_index], img_left);
//             //     } else {
//             //         std::cerr << "Empty mono image at index: " << image_index << std::endl;
//             //     }
//             // }

//             ++image_index;
//             // std::this_thread::sleep_for(std::chrono::milliseconds(1));
//         }
//     }


// private:
//     void load_images(const std::string& left_csv,
//                      const std::string& right_csv,
//                      const std::string& left_img_dir,
//                      const std::string& right_img_dir,
//                      const std::string& img_csv,
//                      std::vector<std::string>& image_left,
//                      std::vector<std::string>& image_right,
//                      std::vector<long double>& timestamps,
//                      std::vector<long double>& imu_time,
//                      std::vector<sensor_msgs::Imu>& imu_data) {
//         // Load left camera data
//         load_single_cam_data(left_csv, left_img_dir, timestamps, image_left);
//         // Load right camera data
//         std::vector<long double> right_timestamps; // Not used but needed for parsing
//         load_single_cam_data(right_csv, right_img_dir, right_timestamps, image_right);

//         // Verify synchronization
//         if (image_left.size() != image_right.size()) {
//             std::cerr << "ERROR: Left/Right image count mismatch ("
//                       << image_left.size() << " vs " << image_right.size() << ")\n";
//             exit(EXIT_FAILURE);
//         }

//         // Preload images to memory
//         image_left_buffer_.reserve(image_left.size());
//         image_right_buffer_.reserve(image_right.size());

//         for (size_t i = 0; i < image_left.size(); ++i) {
//             std::cout<<"path: "<<image_left[i]<<std::endl;
//             cv::Mat left_img = cv::imread(image_left[i], 0);
//             cv::Mat right_img = cv::imread(image_right[i], 0);

//             if (left_img.empty() || right_img.empty()) {
//                 std::cerr << "WARNING: Failed to load image pair " << i << "\n"
//                           << "Left: " << image_left[i] << "\n"
//                           << "Right: " << image_right[i] << std::endl;
//                 continue;
//             }

//             // cv::resize(left_img, left_img,
//             //     cv::Size(),                // 当Size为空时，根据缩放系数计算尺寸
//             //     0.5, 0.5,                  // 宽度和高度缩放系数
//             //     cv::INTER_AREA);           // 推荐用于图像缩小的插值方法
//             // cv::resize(right_img, right_img,
//             //     cv::Size(),                // 当Size为空时，根据缩放系数计算尺寸
//             //     0.5, 0.5,                  // 宽度和高度缩放系数
//             //     cv::INTER_AREA);           // 推荐用于图像缩小的插值方法

//             cv::resize(left_img, left_img, cv::Size(COL, ROW));
//             cv::resize(right_img, right_img, cv::Size(COL, ROW));

//             image_left_buffer_.emplace_back(left_img);
//             image_right_buffer_.emplace_back(right_img);
//             std::cout<<"read image: "<<i<<" in "<<image_left.size()<<std::endl;
//         }

//         std::cout << "Successfully loaded " << image_left_buffer_.size()
//                   << " synchronized image pairs\n";

//         // read imu
//         long double start_time = timestamps[0];
//         // for (auto &t : timestamps) {
//         //     t -= start_time;
//         // }

//         std::ifstream imu_file(img_csv);
//         std::string line;
//         std::getline(imu_file, line); // skip first line
//         while (std::getline(imu_file, line)) {
//             std::istringstream iss(line);
//             std::string token;
//             std::getline(iss, token, ',');
//             // string to long double
//             long double t = std::stod(token);
//             std::getline(iss, token, ',');
//             double wx = std::stod(token);
//             std::getline(iss, token, ',');
//             double wy = std::stod(token);
//             std::getline(iss, token, ',');
//             double wz = std::stod(token);
//             std::getline(iss, token, ',');
//             double ax = std::stod(token);
//             std::getline(iss, token, ',');
//             double ay = std::stod(token);
//             std::getline(iss, token, ',');
//             double az = std::stod(token);
//             sensor_msgs::Imu imu_msg;
//             imu_msg.header.stamp = ros::Time::now();
//             imu_msg.angular_velocity.x = wx;
//             imu_msg.angular_velocity.y = wy;
//             imu_msg.angular_velocity.z = wz;
//             imu_msg.linear_acceleration.x = ax;
//             imu_msg.linear_acceleration.y = ay;
//             imu_msg.linear_acceleration.z = az;
//             imu_data.push_back(imu_msg);
//             imu_time.push_back(t);
//         }

//         // for (auto &t : imu_time) {
//         //     t -= start_time;
//         // }

//     }

//     static void load_single_cam_data(const std::string& csv_path,
//                              const std::string& img_dir,
//                              std::vector<long double>& timestamps,
//                              std::vector<std::string>& img_paths) {

//         std::ifstream file(csv_path);
//         if (!file.is_open()) {
//             std::cerr << "ERROR: Failed to open CSV: " << csv_path << std::endl;
//             exit(EXIT_FAILURE);
//         }

//         // Skip header
//         std::string line;
//         std::getline(file, line);

//         while (std::getline(file, line)) {
//             std::istringstream ss(line);
//             std::string timestamp_str, filename;
//             // ss >> timestamp_str >> filename;
//             // 读取第一个字段（时间戳）
//             std::getline(ss, timestamp_str, ',');
//             // 读取第二个字段（文件名）
//             std::getline(ss, filename, ',');
//             // Convert scientific notation timestamp to seconds
//             timestamps.push_back(std::stod(timestamp_str));

//             img_paths.emplace_back(img_dir + "/" + filename);
//         }
//     }

//     std::mutex img_mutex;
//     std::vector<std::string> image_left_;
//     std::vector<std::string> image_right_;
//     std::vector<long double> image_timestamps_;
//     std::vector<long double> imu_time_;
//     std::vector<sensor_msgs::Imu> imu_data_;
//     std::string dataset_path_;

//     std::vector<cv::Mat> image_left_buffer_;
//     std::vector<cv::Mat> image_right_buffer_;
// };

// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "vins_estimator");
//     ros::NodeHandle n("~");
//     ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

//     if(argc != 3)
//     {
//         printf("please intput: rosrun vins uma_node [config file] /dataset/path \n");
//         return 1;
//     }

//     string config_file = argv[1];
//     printf("config_file: %s\n", argv[1]);

//     string dataset_path = argv[2];

//     readParameters(config_file);
//     estimator.featureTracker.load_model();
//     estimator.setParameter();

// #ifdef EIGEN_DONT_PARALLELIZE
//     ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
// #endif

//     ROS_WARN("waiting for image and imu...");

//     registerPub(n);

//     UMAGrabber grabber(dataset_path);

//     ros::Subscriber sub_imu;
//     if(USE_IMU)
//     {
//         sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
//     }
//     ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
//     ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
//     ros::Subscriber sub_img1;
//     if(STEREO)
//     {
//         sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
//     }
//     ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
//     ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
//     ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);

//     std::thread sensor_thread(&UMAGrabber::sync_process, &grabber);
//     std::thread sync_thread{sync_process};
//     ros::spin();

//     return 0;
// }



#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <fstream>
#include <thread>
#include <opencv2/opencv.hpp>

// #define ROW 384
// #define COL 512

// #define ROW 768
// #define COL 1024

#define ROW 480
#define COL 752

ros::Publisher pub_img_l,pub_img_r;
ros::Publisher pub_imu;

// void readImage(std::string path) {
//     // const std::string path = "/media/linyi/linyicheng/uma-vi/lab-module-csc_2019-02-01-14-28-51_InOut/";
//     // const std::string path = "/media/linyi/linyicheng/uma-vi/lab-module-csc-rev_2019-02-05-17-47-57_InOut/";
//     // const std::string path = "/media/linyi/linyicheng/uma-vi/parking-eng1_2019-02-07-13-38-52_SunOver/";
//     // const std::string path = "/media/linyi/linyicheng/uma-vi/third-floor-csc1_2019-03-04-20-06-37_IllChange/";
//     const std::string timestamp_path = path + "cam2/data.csv";
//     const std::string timestamp_path2 = path + "cam3/data.csv";
//     const std::string imu_path = path + "imu0/data.csv";
//     // read timestamp
//     std::ifstream timestamp_file(timestamp_path);
//     std::string line;
//     std::vector<long double> timestamp;
//     std::vector<std::string> img_name;
//     std::getline(timestamp_file, line); // skip first line
//     while (std::getline(timestamp_file, line)) {
//         std::istringstream iss(line);
//         std::string token;
//         std::getline(iss, token, ',');
//         // string to long double
//         timestamp.push_back(std::stod(token));
//         std::getline(iss, token, ',');
//         img_name.push_back(token);
//     }
//     long double start_time = timestamp[0];
//     // for (auto &t : timestamp) {
//     //     t -= start_time;
//     // }

//     std::vector<std::string> img_name_right;
//     {
//         std::ifstream timestamp_file1(timestamp_path2);
//         std::string line2;
//         std::vector<long double> timestamp2;
//         std::getline(timestamp_file1, line); // skip first line
//         while (std::getline(timestamp_file1, line)) {
//             std::istringstream iss(line);
//             std::string token;
//             std::getline(iss, token, ',');
//             std::getline(iss, token, ',');
//             img_name_right.push_back(token);
//         }
//     }


//     // read image
//     std::vector<cv::Mat> left_images;
//     std::vector<cv::Mat> right_images;
//     for (size_t i = 0; i < img_name.size(); ++i) {
//         cv::Mat img = cv::imread(path + "cam2/data/" + img_name[i], 0);
//         if (img.empty()) {
//             std::cout << "Left image empty: " << img_name[i] << std::endl;
//             // Handle error if needed, e.g., continue or break
//         }
//         cv::resize(img, img, cv::Size(COL, ROW));
//         left_images.push_back(img);

//         cv::Mat img2 = cv::imread(path + "cam3/data/" + img_name_right[i], 0);
//         if (img2.empty()) {
//             std::cout << "Right image empty: " << img_name_right[i] << std::endl;
//             // Handle error if needed
//         }
//         cv::resize(img2, img2, cv::Size(COL, ROW));
//         right_images.push_back(img2);
//     }


//     // read imu
//     std::ifstream imu_file(imu_path);
//     std::vector<sensor_msgs::Imu> imu_data;
//     std::vector<long double> imu_timestamp;
//     std::getline(imu_file, line); // skip first line
//     while (std::getline(imu_file, line)) {
//         std::istringstream iss(line);
//         std::string token;
//         std::getline(iss, token, ',');
//         // string to long double
//         long double t = std::stod(token);
//         std::getline(iss, token, ',');
//         double wx = std::stod(token);
//         std::getline(iss, token, ',');
//         double wy = std::stod(token);
//         std::getline(iss, token, ',');
//         double wz = std::stod(token);
//         std::getline(iss, token, ',');
//         double ax = std::stod(token);
//         std::getline(iss, token, ',');
//         double ay = std::stod(token);
//         std::getline(iss, token, ',');
//         double az = std::stod(token);
//         sensor_msgs::Imu imu_msg;
//         imu_msg.header.stamp = ros::Time::now();
//         imu_msg.angular_velocity.x = wx;
//         imu_msg.angular_velocity.y = wy;
//         imu_msg.angular_velocity.z = wz;
//         imu_msg.linear_acceleration.x = ax;
//         imu_msg.linear_acceleration.y = ay;
//         imu_msg.linear_acceleration.z = az;
//         imu_data.push_back(imu_msg);
//         imu_timestamp.push_back(t);
//     }
//     // for (auto &t : imu_timestamp) {
//     //     t -= start_time;
//     // }
//     // read image and publish
//     // publish first image
//     ros::Time start = ros::Time::now();
//     sensor_msgs::Image img_msg;
//     img_msg.header.stamp = start;
//     img_msg.header.frame_id = "world";
//     img_msg.height = ROW;
//     img_msg.width = COL;
//     img_msg.encoding = "mono8";
//     img_msg.is_bigendian = false;
//     img_msg.step = COL;

//     img_msg.data.resize(ROW * COL);
//     memcpy(&img_msg.data[0], left_images[0].data, ROW * COL);
//     sensor_msgs::Image::ConstPtr img_ptr = boost::make_shared<sensor_msgs::Image>(img_msg);

//     sensor_msgs::Image img_msg2;
//     img_msg2.header.stamp = start;
//     img_msg2.header.frame_id = "world";
//     img_msg2.height = ROW;
//     img_msg2.width = COL;
//     img_msg2.encoding = "mono8";
//     img_msg2.is_bigendian = false;
//     img_msg2.step = COL;

//     img_msg2.data.resize(ROW * COL);
//     memcpy(&img_msg2.data[0], right_images[0].data, ROW * COL);
//     sensor_msgs::Image::ConstPtr img_ptr2 = boost::make_shared<sensor_msgs::Image>(img_msg2);

//     pub_img_l.publish(img_ptr);
//     pub_img_r.publish(img_msg2);

//     int pub_index = 1;
//     int imu_index = 0;

//     while (true) {
//         ros::Time now = ros::Time::now();
//         ros::Duration d = now - start;
//         // d to ns
//         long double d_ns = d.sec * 1e9 + d.nsec;
//         if (d_ns < imu_timestamp[imu_index] - start_time) {
//             continue;
//         }else {
//             // ros::Duration imu_time;
//             // imu_data[imu_index].header.stamp = start + imu_time.fromNSec((int64)imu_timestamp[imu_index]);
//             imu_data[imu_index].header.stamp.sec = static_cast<uint32_t>(imu_timestamp[imu_index] / 1e9);
//             imu_data[imu_index].header.stamp.nsec = static_cast<uint32_t>(fmod(imu_timestamp[imu_index], 1e9));
//             pub_imu.publish(imu_data[imu_index]);
//             imu_index++;
//             if (imu_index == imu_timestamp.size()) {
//                 break;
//             }
//         }
//         if (d_ns < timestamp[pub_index] - start_time) {
//             continue;
//         }else {
//             //
//             ros::Duration img_time;
//             img_msg.header.stamp.sec  = static_cast<uint32_t>(timestamp[pub_index] / 1e9);
//             img_msg.header.stamp.nsec = static_cast<uint32_t>(fmod(timestamp[pub_index], 1e9));

//             img_msg2.header.stamp = img_msg.header.stamp;
  
//             memcpy(&img_msg.data[0], left_images[pub_index].data, ROW * COL);
//             img_ptr = boost::make_shared<sensor_msgs::Image>(img_msg);
//             pub_img_l.publish(img_ptr);

//             memcpy(&img_msg2.data[0], right_images[pub_index].data, ROW * COL);
//             img_ptr2 = boost::make_shared<sensor_msgs::Image>(img_msg2);
//             pub_img_r.publish(img_ptr2);

//             pub_index++;
//             if (pub_index == timestamp.size()) {
//                 break;
//             }
//             std::cout<< "pub_index: " << pub_index <<" time: "<<img_msg.header.stamp.toSec()<< std::endl;
//         }
//     }
// }

void readImage(std::string path) {
    // const std::string path = "/media/linyi/linyicheng/uma-vi/lab-module-csc_2019-02-01-14-28-51_InOut/";
    // const std::string path = "/media/linyi/linyicheng/uma-vi/lab-module-csc-rev_2019-02-05-17-47-57_InOut/";
    // const std::string path = "/media/linyi/linyicheng/uma-vi/parking-eng1_2019-02-07-13-38-52_SunOver/";
    // const std::string path = "/media/linyi/linyicheng/uma-vi/third-floor-csc1_2019-03-04-20-06-37_IllChange/";
    const std::string timestamp_path = path + "cam2/data.csv";
    const std::string timestamp_path2 = path + "cam3/data.csv";
    const std::string imu_path = path + "imu0/data.csv";
    // read timestamp
    std::ifstream timestamp_file(timestamp_path);
    std::string line;
    std::vector<long double> timestamp;
    std::vector<std::string> img_name;
    std::getline(timestamp_file, line); // skip first line
    while (std::getline(timestamp_file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::getline(iss, token, ',');
        // string to long double
        timestamp.push_back(std::stod(token));
        std::getline(iss, token, ',');
        img_name.push_back(token);
    }
    long double start_time = timestamp[0] - 1e9;
    // for (auto &t : timestamp) {
    //     t -= start_time;
    // }

    std::vector<std::string> img_name_right;
    {
        std::ifstream timestamp_file1(timestamp_path2);
        std::string line2;
        std::vector<long double> timestamp2;
        std::getline(timestamp_file1, line); // skip first line
        while (std::getline(timestamp_file1, line)) {
            std::istringstream iss(line);
            std::string token;
            std::getline(iss, token, ',');
            std::getline(iss, token, ',');
            img_name_right.push_back(token);
        }
    }


    // read image


    // read imu
    std::ifstream imu_file(imu_path);
    std::vector<sensor_msgs::Imu> imu_data;
    std::vector<long double> imu_timestamp;
    std::getline(imu_file, line); // skip first line
    while (std::getline(imu_file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::getline(iss, token, ',');
        // string to long double
        long double t = std::stod(token);
        std::getline(iss, token, ',');
        double wx = std::stod(token);
        std::getline(iss, token, ',');
        double wy = std::stod(token);
        std::getline(iss, token, ',');
        double wz = std::stod(token);
        std::getline(iss, token, ',');
        double ax = std::stod(token);
        std::getline(iss, token, ',');
        double ay = std::stod(token);
        std::getline(iss, token, ',');
        double az = std::stod(token);
        sensor_msgs::Imu imu_msg;
        imu_msg.header.stamp = ros::Time::now();
        imu_msg.angular_velocity.x = wx;
        imu_msg.angular_velocity.y = wy;
        imu_msg.angular_velocity.z = wz;
        imu_msg.linear_acceleration.x = ax;
        imu_msg.linear_acceleration.y = ay;
        imu_msg.linear_acceleration.z = az;
        imu_data.push_back(imu_msg);
        imu_timestamp.push_back(t);
    }
    // for (auto &t : imu_timestamp) {
    //     t -= start_time;
    // }
    // read image and publish
    // publish first image
    ros::Time start = ros::Time::now();
    sensor_msgs::Image img_msg;

    img_msg.header.stamp.sec  = static_cast<uint32_t>(timestamp[0] / 1e9);
    img_msg.header.stamp.nsec = static_cast<uint32_t>(fmod(timestamp[0], 1e9));
    img_msg.header.frame_id = "world";
    img_msg.height = ROW;
    img_msg.width = COL;
    img_msg.encoding = "mono8";
    img_msg.is_bigendian = false;
    img_msg.step = COL;

    img_msg.data.resize(ROW * COL);
    cv::Mat img = cv::imread(path +"cam2/data/"+ img_name[0], 0);
    cv::resize(img, img, cv::Size(COL, ROW));
    memcpy(&img_msg.data[0], img.data, ROW * COL);
    sensor_msgs::Image::ConstPtr img_ptr = boost::make_shared<sensor_msgs::Image>(img_msg);

    sensor_msgs::Image img_msg2;
    img_msg2.header.stamp.sec  = static_cast<uint32_t>(timestamp[0] / 1e9);
    img_msg2.header.stamp.nsec = static_cast<uint32_t>(fmod(timestamp[0], 1e9));
    img_msg2.header.frame_id = "world";
    img_msg2.height = ROW;
    img_msg2.width = COL;
    img_msg2.encoding = "mono8";
    img_msg2.is_bigendian = false;
    img_msg2.step = COL;

    img_msg2.data.resize(ROW * COL);
    cv::Mat img2 = cv::imread(path +"cam3/data/"+ img_name_right[0], 0);
    cv::resize(img2, img2, cv::Size(COL, ROW));
    memcpy(&img_msg2.data[0], img2.data, ROW * COL);
    sensor_msgs::Image::ConstPtr img_ptr2 = boost::make_shared<sensor_msgs::Image>(img_msg2);

    pub_img_l.publish(img_ptr);
    pub_img_r.publish(img_msg2);

    int pub_index = 1;
    int imu_index = 0;

    while (true) {
        ros::Time now = ros::Time::now();
        ros::Duration d = now - start;
        // d to ns
        long double d_ns = d.sec * 1e9 + d.nsec;
        if (d_ns < imu_timestamp[imu_index] - start_time) {
            continue;
        }else {
            // ros::Duration imu_time;
            // imu_data[imu_index].header.stamp = start + imu_time.fromNSec((int64)imu_timestamp[imu_index]);
            imu_data[imu_index].header.stamp.sec = static_cast<uint32_t>(imu_timestamp[imu_index] / 1e9);
            imu_data[imu_index].header.stamp.nsec = static_cast<uint32_t>(fmod(imu_timestamp[imu_index], 1e9));
            pub_imu.publish(imu_data[imu_index]);
            imu_index++;
            if (imu_index == imu_timestamp.size()) {
                break;
            }
        }
        if (d_ns < timestamp[pub_index] - start_time) {
            continue;
        }else {
            //
            ros::Duration img_time;
            img_msg.header.stamp.sec  = static_cast<uint32_t>(timestamp[pub_index] / 1e9);
            img_msg.header.stamp.nsec = static_cast<uint32_t>(fmod(timestamp[pub_index], 1e9));

            img_msg2.header.stamp = img_msg.header.stamp;
  
            img = cv::imread(path +"cam2/data/"+ img_name[pub_index], 0);
            img2 = cv::imread(path +"cam3/data/"+ img_name_right[pub_index], 0);

            if (img.empty()) {
                std::cout<<"image empty"<<std::endl;
            }
            cv::resize(img, img, cv::Size(COL, ROW));
            memcpy(&img_msg.data[0], img.data, ROW * COL);
            img_ptr = boost::make_shared<sensor_msgs::Image>(img_msg);
            pub_img_l.publish(img_ptr);


            cv::resize(img2, img2, cv::Size(COL, ROW));
            memcpy(&img_msg2.data[0], img2.data, ROW * COL);
            img_ptr2 = boost::make_shared<sensor_msgs::Image>(img_msg2);
            pub_img_r.publish(img_ptr2);

            pub_index++;
            if (pub_index == timestamp.size()) {
                break;
            }
            std::cout<< "pub_index: " << pub_index <<" time: "<<img_msg.header.stamp.toSec()<< std::endl;
        }
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);


    pub_img_l = n.advertise<sensor_msgs::Image>("/cam0/image_raw", 2000);
    pub_img_r = n.advertise<sensor_msgs::Image>("/cam1/image_raw", 2000);
    pub_imu = n.advertise<sensor_msgs::Imu>("/imu0", 2000);

    // new thread for read image
    std::thread read_img_thread(readImage, argv[1]);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */

    ros::spin();
    return 0;
}

