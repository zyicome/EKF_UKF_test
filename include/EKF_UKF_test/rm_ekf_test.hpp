#include <iostream>
#include <vector>
#include <random>

#include "opencv2/opencv.hpp"

#include "rclcpp/rclcpp.hpp"

class RmEkfTest : public rclcpp::Node
{
public: 
    RmEkfTest();
    void parameters_init();
    cv::Mat control_input();
    cv::Mat observation_model(const cv::Mat &x);
    cv::Mat motion_model(const cv::Mat &x, const cv::Mat &u);
    void observation(cv::Mat &x_true, cv::Mat &z, cv::Mat &xd,cv::Mat &ud);
    cv::Mat jacob_f(const cv::Mat &x, const cv::Mat &u);
    cv::Mat jacob_h();
    void EKF_estimation(cv::Mat &x_est, cv::Mat &P_est, const cv::Mat &z, const cv::Mat &u);
    void EKF_predict(cv::Mat &x_est, cv::Mat &P_est, const cv::Mat &z, const cv::Mat &u);
    void draw(cv::Mat &plotImg, cv::Mat &x_est, cv::Mat &P_est);
    void test();

    double dt;
    double GPS_NOISE;
    double INPUT_NOISE;

    double x;
    double y;
    cv::Point true_last;
    cv::Point true_current;

    double last_x;
    double last_y;
    double last_z;
    double last_yaw;
    cv::Point predict_last;

    double true_x_v;
    double true_x_a;
    double true_y_v;
    double true_y_a;
    double true_z_v;
    double true_z_a;
    double true_yaw_v;
    double true_yaw_a;
    double true_r;


    double x_scale;
    double y_scale;

    cv::Mat Q;
    cv::Mat R;
};