#include "rm_ekf_test.hpp"

RmEkfTest::RmEkfTest() : Node("rm_ekf_test")
{
    RCLCPP_INFO(this->get_logger(), "RmEkfTest node has been created.");
    parameters_init();
    test();
}

void RmEkfTest::parameters_init()
{
    RCLCPP_INFO(this->get_logger(), "Parameters have been initialized.");
    dt = 0.05; // Time step
    GPS_NOISE = 0.1; // GPS measurement noise
    INPUT_NOISE = 0; // Input noise

    x = 0.0;
    y = 0.0;
    true_last = cv::Point(x, y);

    last_x = 0.0;
    last_y = 0.0;
    last_z = 0.0;
    last_yaw = 0.0;
    predict_last = cv::Point(last_x, last_y);

    x_scale = 10;
    y_scale = 10;

    // update_Q - process noise covariance matrix
    double q_v = 1.0;
    double q_vyaw = 1.0;
    double q_r = 1.0;
    double q_v_v = pow(dt, 4) / 4 * q_v;
    double q_v_a = pow(dt, 3) / 2 * q_v;
    double q_a_a = pow(dt, 2) * q_v;
    double q_vyaw_vyaw = pow(dt, 4) / 4 * q_vyaw;
    double q_vyaw_ayaw = pow(dt, 3) / 2 * q_vyaw;
    double q_ayaw_ayaw = pow(dt, 2) * q_vyaw;
    double q_r_r = pow(dt, 4) / 4 * q_r;
    //                              xv     xa          yv     ya          zv     za          vyaw     ayaw         r 
    Q = (cv::Mat_<double>(9,9) << q_v_v, q_v_a, 0, 0, 0, 0, 0, 0, 0,
                                  q_v_a, q_a_a, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, q_v_v, q_v_a, 0, 0, 0, 0, 0,
                                  0, 0, q_v_a, q_a_a, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, q_v_v, q_v_a, 0, 0, 0,
                                  0, 0, 0, 0, q_v_a, q_a_a, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, q_vyaw_vyaw, q_vyaw_ayaw, 0,
                                  0, 0, 0, 0, 0, 0, q_vyaw_ayaw, q_ayaw_ayaw, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, q_r_r);

    // update_R - measurement noise covariance matrix
    double r_v = 0.05;
    double r_vyaw = 0.02;
    R = (cv::Mat_<double>(4, 4) << r_v, 0, 0, 0,
                                   0, r_v, 0, 0,
                                   0, 0, r_v, 0,
                                   0, 0, 0, r_vyaw);

    true_x_v = 2.5; // m/s
    true_y_v = 0.5; // m/s
    true_z_v = 0.5; // m/s
    true_yaw_v = 0.01; // rad/s
    true_r = 25; // m
    true_x_a = 0.05; // m/s^2
    true_y_a = 0.05; // m/s^2
    true_z_a = 0.05; // m/s^2
    true_yaw_a = 0.05; // rad/s^2
    RCLCPP_INFO(this->get_logger(), "Finished init parameters.");
}

cv::Mat RmEkfTest::control_input()
{
    true_x_a += 0.001; // m/s^2
    true_y_a += 0.001; // m/s^2
    true_z_a += 0.001; // m/s^2
    true_yaw_a += 0.001; // rad/s^2
    true_x_v = true_x_v + true_x_a * dt;
    true_y_v = true_y_v + true_y_a * dt;
    true_z_v = true_z_v + true_z_a * dt;
    true_yaw_v = true_yaw_v + true_yaw_a * dt;
    cv::Mat u = (cv::Mat_<double>(4, 1) << true_x_v, true_y_v, true_z_v, true_yaw_v);
    return u;
}

cv::Mat RmEkfTest::observation_model(const cv::Mat &x)
{
    cv::Mat H = (cv::Mat_<double>(4, 9) << 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 1, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 1, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 1, 0, 0);
    cv::Mat z = H * x;
    return z;
}

cv::Mat RmEkfTest::motion_model(const cv::Mat &x, const cv::Mat &u)
{
    cv::Mat F = (cv::Mat_<double>(9, 9) << 1, dt, 0, 0, 0, 0, 0, 0, 0,
                                           0, 1, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 1, dt, 0, 0, 0, 0, 0,
                                           0, 0, 0, 1, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 1, dt, 0, 0, 0,
                                           0, 0, 0, 0, 0, 1, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 1, dt, 0,
                                           0, 0, 0, 0, 0, 0, 0, 1, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 1);
    /*cv::Mat F = (cv::Mat_<double>(9, 9) << 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 1, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 1, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 1, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 1, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 1, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 1, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 1, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 1);*/
    /*cv::Mat B = (cv::Mat_<double>(9, 4) << dt, 0, 0, 0,
                                           0, dt, 0, 0,
                                           0, 0, dt, 0,
                                           0, 0, 0, dt,
                                           dt, 0, 0, 0,
                                           0, dt, 0, 0,
                                           0, 0, dt, 0,
                                           0, 0, 0, dt,
                                           0, 0, 0, 0); // ???*/
    //cv::Mat x_dot = F * x + B * u;
    cv::Mat x_dot = F * x;
    return x_dot;
}

void RmEkfTest::observation(cv::Mat &x_true, cv::Mat &z, cv::Mat &xd,cv::Mat &ud)
{
    std::random_device rd;  // 用于获得一个随机种子
    std::mt19937 gen(rd()); // 以随机设备rd初始化Mersenne Twister生成器
    // 定义随机数分布，这里是在1到10之间的均匀分布
    std::uniform_int_distribution<> distrib(1, 2);

    x_true = motion_model(x_true, ud);
    //add noise to gps x-y
    z = observation_model(x_true) + GPS_NOISE * (cv::Mat_<double>(4, 1) << distrib(gen), distrib(gen), distrib(gen), distrib(gen));

    // add noise to input
    ud = ud + INPUT_NOISE * (cv::Mat_<double>(4, 1) << distrib(gen), distrib(gen), distrib(gen), distrib(gen));

    // update state with noised input
    xd = motion_model(xd, ud);
}

cv::Mat RmEkfTest::jacob_f(const cv::Mat &x, const cv::Mat &u)
{
    cv::Mat jF = (cv::Mat_<double>(9, 9) << 1, dt, 0, 0, 0, 0, 0, 0, 0,
                                            0, 1, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 1, dt, 0, 0, 0, 0, 0,
                                            0, 0, 0, 1, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 1, dt, 0, 0, 0,
                                            0, 0, 0, 0, 0, 1, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 1, dt, 0,
                                            0, 0, 0, 0, 0, 0, 0, 1, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 1);
    return jF;
}

cv::Mat RmEkfTest::jacob_h()
{
    cv::Mat jH = (cv::Mat_<double>(4, 9) << 1,   0,   0,   0,   0,   0,   0,          0,   0,
                                            0,   0,   1,   0,   0,   0,   0,          0,   0,
                                            0,   0,   0,   0,   1,   0,   0,          0,   0,
                                            0,   0,   0,   0,   0,   0,   1,          0,   0);
    return jH;  
}

void RmEkfTest::EKF_predict(cv::Mat &x_est, cv::Mat &P_est, const cv::Mat &z, const cv::Mat &u)
{
    // 1. Predict
    cv::Mat x_pred = motion_model(x_est, u);
    cv::Mat jF = jacob_f(x_est, u);
    cv::Mat P_pred = jF * P_est * jF.t() + Q;
    x_est = x_pred;
    P_est = P_pred;
}

void RmEkfTest::EKF_estimation(cv::Mat &x_est, cv::Mat &P_est, const cv::Mat &z, const cv::Mat &u)
{
    // 1. Predict
    cv::Mat x_pred = motion_model(x_est, u);
    cv::Mat jF = jacob_f(x_est, u);
    cv::Mat P_pred = jF * P_est * jF.t() + Q;

    // 2. Update
    cv::Mat jH = jacob_h();
    cv::Mat z_pred = observation_model(x_pred);
    cv::Mat y = z - z_pred;
    cv::Mat S = jH * P_pred * jH.t() + R;
    cv::Mat K = P_pred * jH.t() * S.inv(cv::DECOMP_SVD);
    x_est = x_pred + K * y;
    P_est = (cv::Mat::eye(x_est.rows, x_est.rows, CV_64F) - K * jH) * P_pred;
}

void RmEkfTest::draw(cv::Mat &plotImg, cv::Mat &x_est, cv::Mat &P_est)
{
    std::cout << "x_est: " << x_est << std::endl;
    double x_v = x_est.at<double>(0, 0);
    double x_a = x_est.at<double>(1, 0);
    double y_v = x_est.at<double>(2, 0);
    double y_a = x_est.at<double>(3, 0);
    double z_v = x_est.at<double>(4, 0);
    double z_a = x_est.at<double>(5, 0);
    double yaw_v = x_est.at<double>(6, 0);
    double yaw_a = x_est.at<double>(7, 0);
    double r = x_est.at<double>(8, 0);
    r = 25;

    double x = last_x + x_v * dt + 0.5 * x_a * dt * dt;
    double y = last_y + y_v * dt + 0.5 * y_a * dt * dt;
    double z = last_z + z_v * dt + 0.5 * z_a * dt * dt;
    double yaw = last_yaw + yaw_v * dt + 0.5 * yaw_a * dt * dt;

    std::cout << "x: " << x << " y: " << y << " z: " << z << " yaw: " << yaw << std::endl;
    std::cout << "x_v: " << x_v << std::endl;
    std::cout << "x_a: " << x_a << std::endl;
    std::cout << "y_v: " << y_v << std::endl;
    std::cout << "y_a: " << y_a << std::endl;
    std::cout << "z_v: " << z_v << std::endl;
    std::cout << "z_a: " << z_a << std::endl;
    std::cout << "yaw_v: " << yaw_v << std::endl;
    std::cout << "yaw_a: " << yaw_a << std::endl;
    std::cout << "r: " << r << std::endl;

    cv::Point predict_current = cv::Point(x, y);

    cv::line(plotImg, predict_last, predict_current, cv::Scalar(0, 255, 0), 3);

    double x_yaw = x + r * cos(yaw);
    double y_yaw = y + r * sin(yaw);
    /*cv::line(plotImg, predict_current, cv::Point(x_yaw, y_yaw), cv::Scalar(0, 0, 255), 1);
    cv::circle(plotImg, cv::Point(x_yaw, y_yaw), 1, cv::Scalar(0, 0, 255), -1);*/

    predict_last = predict_current;
    last_x = x;
    last_y = y;
    last_z = z;
    last_yaw = yaw;
}

void RmEkfTest::test()
{
    const int nx = 9; // State vector size
    cv::Mat xEst = cv::Mat::zeros(nx, 1, CV_64F);
    cv::Mat xTrue = cv::Mat::zeros(nx, 1, CV_64F);
    cv::Mat PEst = cv::Mat::eye(nx, nx, CV_64F);
    cv::Mat xDR = cv::Mat::zeros(nx, 1, CV_64F); // Dead reckoning

    cv::Mat z = cv::Mat::zeros(4, 1, CV_64F); // Observation vector

    // History containers for plotting
    std::vector<cv::Point2f> hxEst, hxTrue, hxDR, hz;

    double time = 0.0;
    const double SIM_TIME = 60.0; // Simulation time

    // Setup OpenCV window for plotting
    cv::namedWindow("EKF Simulation", cv::WINDOW_AUTOSIZE);
    cv::Mat plotImg;
    // Plotting (simplified, assuming functions for trajectory and covariance ellipse plotting)
    plotImg = cv::Mat::zeros(1500, 1500, CV_8UC3) + cv::Scalar(255, 255, 255);

    while (time <= SIM_TIME)
    {
        if(time <= 60)
        {
            time += dt;
            auto u = control_input();
            xTrue.at<double>(0) = u.at<double>(0);
            xTrue.at<double>(2) = u.at<double>(1);
            xTrue.at<double>(4) = u.at<double>(2);
            xTrue.at<double>(6) = u.at<double>(3);
            observation(xTrue, z, xDR, u);
            EKF_estimation(xEst, PEst, z, u);
            std::cout << "4" << std::endl;

            std::cout << "xTrue: " << xTrue << std::endl;
            std::cout << "z: " << z << std::endl;
            std::cout << "xEst: " << xEst << std::endl;
            std::cout << "PEst: " << PEst << std::endl;
            std::cout << "xDR: " << xDR << std::endl;
            std::cout << "u: " << u << std::endl;
            std::cout << "q: " << Q << std::endl;
            std::cout << "r: " << R << std::endl;

            // Store data history for plotting
            // Note: Conversion from cv::Mat to cv::Point2f for plotting
            hxEst.push_back(cv::Point2f(xEst.at<double>(0), xEst.at<double>(1)));
            hxDR.push_back(cv::Point2f(xDR.at<double>(0), xDR.at<double>(1)));
            hxTrue.push_back(cv::Point2f(xTrue.at<double>(0), xTrue.at<double>(1)));
            hz.push_back(cv::Point2f(z.at<double>(0), z.at<double>(1)));

            //plot_trajectories(plotImg, hxTrue, hxEst, hxDR)

            x = x + true_x_v * dt + 0.5 * true_x_a * dt * dt;
            y = y + true_y_v * dt + 0.5 * true_y_a * dt * dt;
            true_current = cv::Point(x, y);
            cv::line(plotImg, true_last, true_current, cv::Scalar(0, 0, 0), 2);
            true_last = true_current;

            draw(plotImg, xEst, PEst);

            cv::imshow("EKF Simulation", plotImg);
            char key = (char)cv::waitKey(1);
            if(key == 27)
            {
                break;
            }
        }
        else
        {
            time += dt;
            auto u = control_input();
            observation(xTrue, z, xDR, u);
            EKF_predict(xEst, PEst, z, u);

            x = x + true_x_v * dt + 0.5 * true_x_a * dt * dt;
            y = y + true_y_v * dt + 0.5 * true_y_a * dt * dt;
            true_current = cv::Point(x, y);
            cv::line(plotImg, true_last, true_current, cv::Scalar(0, 0, 0), 2);
            true_last = true_current;

            draw(plotImg, xEst, PEst);
            
            cv::imshow("EKF Simulation", plotImg);
            char key = (char)cv::waitKey(1);
            if(key == 27)
            {
                break;
            }
        }
        std::cout << "true_x_v: " << true_x_v << std::endl;
        std::cout << "true_x_a: " << true_x_a << std::endl;
        std::cout << "true_y_v: " << true_y_v << std::endl;
        std::cout << "true_y_a: " << true_y_a << std::endl;
        std::cout << "true_z_v: " << true_z_v << std::endl;
        std::cout << "true_z_a: " << true_z_a << std::endl;
        std::cout << "true_yaw_v: " << true_yaw_v << std::endl;
        std::cout << "true_yaw_a: " << true_yaw_a << std::endl;
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RmEkfTest>());
    rclcpp::shutdown();
    return 0;
}