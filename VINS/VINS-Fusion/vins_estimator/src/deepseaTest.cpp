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


#define ROW 608
#define COL 968

ros::Publisher pub_img_l,pub_img_r;
ros::Publisher pub_imu;


void readImage(std::string path) {

    const std::string timestamp_path = path + "raw_data/img_sequence_2.csv";
    const std::string imu_path = path + "raw_data/imu_sequence_2.csv";
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
    cv::Mat img = cv::imread(path +"raw_data/images_sequence_2/"+ img_name[0], 0);
    cv::resize(img, img, cv::Size(COL, ROW));
    memcpy(&img_msg.data[0], img.data, ROW * COL);
    sensor_msgs::Image::ConstPtr img_ptr = boost::make_shared<sensor_msgs::Image>(img_msg);
    pub_img_l.publish(img_ptr);

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

            img = cv::imread(path +"raw_data/images_sequence_2/"+ img_name[pub_index], 0);

            if (img.empty()) {
                std::cout<<"image empty"<<std::endl;
            }

            cv::resize(img, img, cv::Size(COL, ROW));
            memcpy(&img_msg.data[0], img.data, ROW * COL);
            img_ptr = boost::make_shared<sensor_msgs::Image>(img_msg);
            pub_img_l.publish(img_ptr);

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

