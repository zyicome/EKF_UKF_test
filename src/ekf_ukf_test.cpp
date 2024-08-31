#include "ekf_ukf_test.hpp"

EKF_UKF_test::EKF_UKF_test() : Node("ekf_ukf_test")
{
    RCLCPP_INFO(this->get_logger(), "EKF_UKF_test node has been created.");
    EKF_init();
    EKF_test();
}

void EKF_UKF_test::EKF_init()
{
    RCLCPP_INFO(this->get_logger(), "EKF has been initialized.");
    // Convert degrees to radians for the yaw angle variance
    double yaw_variance_radians = (CV_PI / 180.0);
    // Initialize the Q matrix with the specified variances
    Q = (cv::Mat_<double>(4, 4) << 0.1, 0, 0, 0,
                                           0, 0.1, 0, 0,
                                           0, 0, yaw_variance_radians, 0,
                                           0, 0, 0, 1.0);
    // Square the matrix
    Q = Q.mul(Q);
    // Optionally, print the matrix to verify
    std::cout << "Q Matrix: " << std::endl << Q << std::endl;

    // Initialize the R matrix with the specified variances
    R = (cv::Mat_<double>(2, 2) << 1.0, 0,
                                           0, 1.0);
    // Square the matrix
    R = R.mul(R);
    // Optionally, print the matrix to verify
    std::cout << "R Matrix: " << std::endl << R << std::endl;

    dt = 0.1; // Time step
    GPS_NOISE = 0.0; // GPS measurement noise
    INPUT_NOISE = 0.0; // Input noise
}

cv::Mat EKF_UKF_test::control_input()
{
    //RCLCPP_INFO(this->get_logger(), "EKF has been updated.");
    double v = 0.5; // m/s
    double yaw_rate = 0.05; // rad/s
    cv::Mat u = (cv::Mat_<double>(2, 1) << v, 
                                        yaw_rate);
    return u;
}

cv::Mat EKF_UKF_test::observation_model(const cv::Mat &x)
{
    cv::Mat H = (cv::Mat_<double>(2, 4) << 1, 0, 0, 0,
                                           0, 1, 0, 0);
    cv::Mat z = H * x;
    return z;
}

cv::Mat EKF_UKF_test::motion_model(const cv::Mat &x, const cv::Mat &u)
{
    cv::Mat F = (cv::Mat_<double>(4, 4) << 1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1, 0,
                                           0, 0, 0, 0);
    cv::Mat B = (cv::Mat_<double>(4, 2) << dt * std::cos(x.at<double>(2, 0)), 0,
                                           dt *std::sin(x.at<double>(2, 0)), 0,
                                            0, dt,
                                            1.0, 0);
    cv::Mat x_dot = F * x + B * u;
    return x_dot;
}

void EKF_UKF_test::observation(cv::Mat &x_true, cv::Mat &z, cv::Mat &xd,cv::Mat &ud)
{
    x_true = motion_model(x_true, ud);
    //add noise to gps x-y
    z = observation_model(x_true) + GPS_NOISE ;

    // add noise to input
    ud = ud + INPUT_NOISE;

    // update state with noised input
    xd = motion_model(xd, ud);
}

cv::Mat EKF_UKF_test::jacob_f(const cv::Mat &x, const cv::Mat &u)
{
    /*Jacobian of Motion Model
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)*/
    double yaw = x.at<double>(2, 0);
    double v = u.at<double>(0, 0);
    cv::Mat jF = (cv::Mat_<double>(4, 4) << 1, 0, -dt * v * std::sin(yaw), dt * std::cos(yaw),
                                            0, 1, dt * v * std::cos(yaw), dt * std::sin(yaw),
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);
    return jF;
}

cv::Mat EKF_UKF_test::jacob_h()
{
    //Jacobian of Observation Model
    cv::Mat jH = (cv::Mat_<double>(2, 4) << 1, 0, 0, 0,
                                            0, 1, 0, 0);
    return jH;
}

void EKF_UKF_test::EKF_estimation(cv::Mat &x_est, cv::Mat &P_est, const cv::Mat &z, const cv::Mat &u)
{
    /*两个阶段，预测和更新

    Args:
        x_est (_type_): 估计的状态
        P_est (_type_): 估计的P矩阵（后验估计误差协方差矩阵）
        z (_type_): 观测向量
        u (_type_): 控制输入*/
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

void EKF_UKF_test::draw(cv::Mat &plotImg, cv::Mat &x_est, cv::Mat &P_est)
{
    // 1. Extract the top-left 2x2 submatrix of P_est
    cv::Mat Pxy = P_est(cv::Rect(0, 0, 2, 2));

    // 2. Compute the eigenvalues and eigenvectors
    cv::Mat eigval, eigvec;
    cv::eigen(Pxy, eigval, eigvec);

    // 3. Determine the indices of the largest and smallest eigenvalues
    int bigind = eigval.at<double>(0, 0) >= eigval.at<double>(1, 0) ? 0 : 1;
    int smallind = bigind == 0 ? 1 : 0;

    // 4. Generate ellipse points
    std::vector<cv::Point> ellipsePoints;
    double t = 0;
    double a = std::sqrt(eigval.at<double>(bigind, 0));
    double b = std::sqrt(eigval.at<double>(smallind, 0));
    double angle = std::atan2(eigvec.at<double>(1, bigind), eigvec.at<double>(0, bigind));
    for (t = 0; t <= 2 * CV_PI + 0.1; t += 0.1) {
        double x = a * std::cos(t);
        double y = b * std::sin(t);

        // Rotate points
        double xRot = x * std::cos(angle) - y * std::sin(angle);
        double yRot = x * std::sin(angle) + y * std::cos(angle);

        // Translate points
        xRot += x_est.at<double>(0, 0);
        xRot = xRot * 10;
        yRot += x_est.at<double>(1, 0);
        yRot = yRot * 10;

        std::cout << "xRot: " << xRot << " yRot: " << yRot << std::endl;
        std::cout << "a: " << a << " b: " << b << "t" << t << std::endl;
        std::cout << "x_est.at<double>(0, 0): " << x_est.at<double>(0, 0) << " x_est.at<double>(1, 0): " << x_est.at<double>(1, 0) << std::endl;
        std::cout << "x: " << x << " y: " << y << std::endl;
        std::cout << "angle: " << angle << std::endl;
        /*std::cout << "x: " << x << " y: " << y << std::endl;
        std::cout << "a: " << a << " b: " << b << "t" << t << std::endl;*/
        ellipsePoints.push_back(cv::Point(xRot, yRot));
    }

    cv::polylines(plotImg, ellipsePoints, true, cv::Scalar(0, 0, 255), 2);

    /*std::cout << "rotatedEllipsePoints.size(): " << rotatedEllipsePoints.size() << std::endl;
    std::cout << "ellipsePoints.size(): " << ellipsePoints.size() << std::endl;
    for(int i = 0; i < rotatedEllipsePoints.size(); i++)
    {
        std::cout << "rotatedEllipsePoints[" << i << "]: " << rotatedEllipsePoints[i] << std::endl;
    }
    for(int i = 0; i < ellipsePoints.size(); i++)
    {
        std::cout << "ellipsePoints[" << i << "]: " << ellipsePoints[i] << std::endl;
    }*/

    
    //cv::imshow("Ellipse", plotImg);
    //cv::waitKey(0);
}

void EKF_UKF_test::EKF_test()
{
    const int nx = 4; // State vector size
    cv::Mat xEst = cv::Mat::zeros(nx, 1, CV_64F);
    cv::Mat xTrue = cv::Mat::zeros(nx, 1, CV_64F);
    cv::Mat PEst = cv::Mat::eye(nx, nx, CV_64F);
    cv::Mat xDR = cv::Mat::zeros(nx, 1, CV_64F); // Dead reckoning

    cv::Mat z = cv::Mat::zeros(2, 1, CV_64F); // Observation vector

    // History containers for plotting
    std::vector<cv::Point2f> hxEst, hxTrue, hxDR, hz;

    double time = 0.0;
    const double SIM_TIME = 60.0; // Simulation time

    // Setup OpenCV window for plotting
    cv::namedWindow("EKF Simulation", cv::WINDOW_AUTOSIZE);
    cv::Mat plotImg;

    while (time <= SIM_TIME) {
        time += dt;
        auto u = control_input();
        observation(xTrue, z, xDR, u);
        EKF_estimation(xEst, PEst, z, u);

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

        // Plotting (simplified, assuming functions for trajectory and covariance ellipse plotting)
        plotImg = cv::Mat::zeros(600, 800, CV_8UC3) + cv::Scalar(255, 255, 255);
        //plot_trajectories(plotImg, hxTrue, hxEst, hxDR);
        draw(plotImg, xEst, PEst);

        cv::imshow("EKF Simulation", plotImg);
        char key = (char)cv::waitKey(1);
        if (key == 27) break; // ESC to exit
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKF_UKF_test>());
    rclcpp::shutdown();
    return 0;
}