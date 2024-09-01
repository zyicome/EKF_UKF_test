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

    double v;
    double yaw_rate;

    double x;
    double y;

    double x_scale;
    double y_scale;

    cv::Point true_last;
    cv::Point true_current;

    cv::Mat Q;
    cv::Mat R;

    //-------------------------------------------
    // UKF
    double ALPHA;
    double BETA;
    double KAPPA;

    void UKF_init();
    void setup_ukf(double nx, cv::Mat &wm, cv::Mat &wc, double &gamma);
    cv::Mat generateSigmaPoints(const cv::Mat &xEst, const cv::Mat &PEst, double gamma);
    cv::Mat predictSigmaMotion(cv::Mat &sigma, const cv::Mat &u);
    cv::Mat predictSigmaObservation(cv::Mat &sigma);
    cv::Mat calcSigmaCovariance(const cv::Mat& x, const cv::Mat& sigma, const cv::Mat& wc, const cv::Mat& Pi);
    cv::Mat calcPxz(const cv::Mat& sigma, const cv::Mat& x, const cv::Mat& zSigma, const cv::Mat& zb, const cv::Mat& wc);
    void UKF_estimation(cv::Mat &xEst, cv::Mat &PEst, const cv::Mat &z, const cv::Mat &u, const cv::Mat &wm, const cv::Mat &wc, double gamma, const cv::Mat &Q, const cv::Mat &R);
    void UKF_test();
};