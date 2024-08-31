#include <iostream>
#include <vector>
#include <random>

#include "opencv2/opencv.hpp"

#include "rclcpp/rclcpp.hpp"

class EKF_UKF_test : public rclcpp::Node
{
public:
    EKF_UKF_test();

    void EKF_init();
    cv::Mat control_input();
    cv::Mat observation_model(const cv::Mat &x);
    cv::Mat motion_model(const cv::Mat &x, const cv::Mat &u);
    void observation(cv::Mat &x_true, cv::Mat &z, cv::Mat &xd,cv::Mat &ud);
    cv::Mat jacob_f(const cv::Mat &x, const cv::Mat &u);
    cv::Mat jacob_h();
    void EKF_estimation(cv::Mat &x_est, cv::Mat &P_est, const cv::Mat &z, const cv::Mat &u);
    void draw(cv::Mat &plotImg, cv::Mat &x_est, cv::Mat &P_est);
    void EKF_test();

    double dt;
    double GPS_NOISE;
    double INPUT_NOISE;

    cv::Mat Q;
    cv::Mat R;
};