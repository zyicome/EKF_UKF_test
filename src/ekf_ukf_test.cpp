#include "ekf_ukf_test.hpp"

EKF_UKF_test::EKF_UKF_test() : Node("ekf_ukf_test")
{
    RCLCPP_INFO(this->get_logger(), "EKF_UKF_test node has been created.");
    EKF_init();
    EKF_test();
    //UKF_init();
    //UKF_test();
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

    dt = 0.05; // Time step
    GPS_NOISE = 0.5; // GPS measurement noise
    INPUT_NOISE = 0.25; // Input noise

    x = 0.0;
    y = 0.0;
    true_last = cv::Point(x, y);

    x_scale = 10;
    y_scale = 10;

}

cv::Mat EKF_UKF_test::control_input()
{
    //RCLCPP_INFO(this->get_logger(), "EKF has been updated.");
    v = 0.5; // m/s
    yaw_rate = 0.05; // rad/s
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
    std::random_device rd;  // 用于获得一个随机种子
    std::mt19937 gen(rd()); // 以随机设备rd初始化Mersenne Twister生成器
    // 定义随机数分布，这里是在1到10之间的均匀分布
    std::uniform_int_distribution<> distrib(1, 2);

    x_true = motion_model(x_true, ud);
    //add noise to gps x-y
    z = observation_model(x_true) + GPS_NOISE * (cv::Mat_<double>(2, 1) << distrib(gen), distrib(gen));

    // add noise to input
    ud = ud + INPUT_NOISE * (cv::Mat_<double>(2, 1) << distrib(gen), distrib(gen));

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
        xRot = xRot * x_scale;
        yRot += x_est.at<double>(1, 0);
        yRot = yRot * y_scale;

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
    // Plotting (simplified, assuming functions for trajectory and covariance ellipse plotting)
    plotImg = cv::Mat::zeros(600, 800, CV_8UC3) + cv::Scalar(255, 255, 255);

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

        //plot_trajectories(plotImg, hxTrue, hxEst, hxDR);
        draw(plotImg, xEst, PEst);

        x = x + v * std::cos(yaw_rate * time) * dt * x_scale;    
        y = y + v * std::sin(yaw_rate * time) * dt * y_scale;
        true_current = cv::Point(x, y);
        cv::line(plotImg, true_last, true_current, cv::Scalar(0, 0, 0), 2);
        true_last = true_current;

        cv::imshow("EKF Simulation", plotImg);
        char key = (char)cv::waitKey(1);
        if (key == 27) break; // ESC to exit
    }
}

void EKF_UKF_test::UKF_init()
{
    RCLCPP_INFO(this->get_logger(), "UKF has been initialized.");
    ALPHA = 0.001;
    BETA = 2;
    KAPPA = 0;

    //QR与EKF相同，不重复初始化了
}

void EKF_UKF_test::setup_ukf(double nx, cv::Mat &wm, cv::Mat &wc, double &gamma)
{
    double lamb = ALPHA * ALPHA * (nx + KAPPA) - nx;
    // Initialize weight matrices
    wm = cv::Mat::zeros(1, 2 * static_cast<int>(nx) + 1, CV_64F);
    wc = cv::Mat::zeros(1, 2 * static_cast<int>(nx) + 1, CV_64F);

    // Calculate weights
    wm.at<double>(0, 0) = lamb / (lamb + nx);
    wc.at<double>(0, 0) = wm.at<double>(0, 0) + (1 - std::pow(ALPHA, 2) + BETA);

    for (int i = 1; i < 2 * nx + 1; ++i) {
        wm.at<double>(0, i) = 1.0 / (2 * (nx + lamb));
        wc.at<double>(0, i) = 1.0 / (2 * (nx + lamb));
    }
    
    // Calculate gamma
    gamma = std::sqrt(nx + lamb);
}

// control_input以及observation_model以及motion_model以及observation函数不变

cv::Mat EKF_UKF_test::generateSigmaPoints(const cv::Mat &xEst, const cv::Mat &PEst, double gamma)
{
    // Ensure xEst is a column vector
    CV_Assert(xEst.cols == 1);

    int n = xEst.rows;
    cv::Mat sigma = xEst.clone();

    // Calculate the square root of PEst using eigenvalue decomposition
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(PEst, eigenvalues, eigenvectors);

    // Compute the square root of the eigenvalues matrix
    cv::Mat sqrtEigenvalues = eigenvalues.clone();
    for (int i = 0; i < eigenvalues.rows; i++) {
        sqrtEigenvalues.at<double>(i, 0) = std::sqrt(eigenvalues.at<double>(i, 0));
    }

    // Compute the square root of PEst
    cv::Mat Psqrt = eigenvectors * cv::Mat::diag(sqrtEigenvalues) * eigenvectors.t();

    // Positive direction
    for (int i = 0; i < n; i++) {
        cv::Mat scaledColumn = gamma * Psqrt.col(i);
        cv::Mat positiveSigmaPoint = xEst + scaledColumn;
        cv::hconcat(sigma, positiveSigmaPoint, sigma); // Correct usage
    }

    // Negative direction
    for (int i = 0; i < n; i++) {
        cv::Mat scaledColumn = gamma * Psqrt.col(i);
        cv::Mat negativeSigmaPoint = xEst - scaledColumn;
        cv::hconcat(sigma, negativeSigmaPoint, sigma); // Correct usage
    }

    return sigma;
}

cv::Mat EKF_UKF_test::predictSigmaMotion(cv::Mat &sigma, const cv::Mat &u) {
    cv::Mat sigmaPred = sigma.clone();
    for (int i = 0; i < sigmaPred.cols; ++i) {
        cv::Mat sigmaCol = sigmaPred.col(i).clone(); // Extract the i-th column
        cv::Mat predictedState = motion_model(sigmaCol, u); // Apply the motion model
        predictedState.copyTo(sigmaPred.col(i)); // Update the i-th column with the predicted state
    }
    return sigmaPred;
}

cv::Mat EKF_UKF_test::predictSigmaObservation(cv::Mat &sigma) {
    cv::Mat sigmaObs = sigma.clone(); // Clone sigma to preserve the original
    for (int i = 0; i < sigmaObs.cols; ++i) {
        cv::Mat sigmaCol = sigmaObs.col(i).clone(); // Extract the i-th column
        cv::Mat observedState = observation_model(sigmaCol); // Apply the observation model
        // Update the first two rows of the i-th column with the observed state
        observedState.row(0).copyTo(sigmaObs.row(0).col(i));
        observedState.row(1).copyTo(sigmaObs.row(1).col(i));
    }
    // Keep only the first two rows
    sigmaObs = sigmaObs.rowRange(0, 2);
    return sigmaObs;
}

cv::Mat EKF_UKF_test::calcSigmaCovariance(const cv::Mat& x, const cv::Mat& sigma, const cv::Mat& wc, const cv::Mat& Pi) {
    int nSigma = sigma.cols;
    cv::Mat P = Pi.clone(); // Clone Pi to ensure we're not modifying the original
    for (int i = 0; i < nSigma; ++i) {
        cv::Mat d = sigma.col(i) - x.rowRange(0, sigma.rows); // Subtract mean from sigma point
        cv::Mat d_outer = d * d.t(); // Calculate the outer product
        P += wc.at<double>(0, i) * d_outer; // Scale by weight and add to covariance
    }
    return P;
}

cv::Mat EKF_UKF_test::calcPxz(const cv::Mat& sigma, const cv::Mat& x, const cv::Mat& zSigma, const cv::Mat& zb, const cv::Mat& wc) {
    int nSigma = sigma.cols;
    cv::Mat dx = sigma - cv::repeat(x, 1, sigma.cols); // Subtract state mean from each sigma point
    cv::Mat dz = zSigma - cv::repeat(zb.rowRange(0, 2), 1, zSigma.cols); // Subtract measurement mean from each measurement sigma point
    cv::Mat P = cv::Mat::zeros(dx.rows, dz.rows, CV_64F); // Initialize cross-covariance matrix

    for (int i = 0; i < nSigma; ++i) {
        cv::Mat dx_col = dx.col(i);
        cv::Mat dz_col = dz.col(i);
        P += wc.at<double>(0, i) * dx_col * dz_col.t(); // Scale and add to cross-covariance
    }

    return P;
}

void EKF_UKF_test::UKF_estimation(cv::Mat &xEst, cv::Mat &PEst, const cv::Mat &z, const cv::Mat &u, const cv::Mat &wm, const cv::Mat &wc, double gamma, const cv::Mat &Q, const cv::Mat &R) {
    // Predict
    cv::Mat sigma = generateSigmaPoints(xEst, PEst, gamma);
    sigma = predictSigmaMotion(sigma, u);
    cv::Mat xPred = cv::Mat::zeros(xEst.rows, 1, CV_64F);
    for (int i = 0; i < sigma.cols; ++i) {
        xPred += wm.at<double>(0, i) * sigma.col(i);
    }
    cv::Mat PPred = calcSigmaCovariance(xPred, sigma, wc, Q);

    // Update
    sigma = generateSigmaPoints(xPred, PPred, gamma);
    cv::Mat zSigma = predictSigmaObservation(sigma);
    cv::Mat zb = cv::Mat::zeros(z.rows, 1, CV_64F);
    for (int i = 0; i < zSigma.cols; ++i) {
        zb += wm.at<double>(0, i) * zSigma.col(i);
    }
    cv::Mat st = calcSigmaCovariance(zb, zSigma, wc, R);
    cv::Mat Pxz = calcPxz(sigma, xPred, zSigma, zb, wc);
    cv::Mat K = Pxz * st.inv();
    cv::Mat y = z - zb;
    xEst = xPred + K * y;
    PEst = PPred - K * st * K.t();
}

void EKF_UKF_test::UKF_test()
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
    cv::namedWindow("UKF Simulation", cv::WINDOW_AUTOSIZE);
    cv::Mat plotImg;
    // Plotting (simplified, assuming functions for trajectory and covariance ellipse plotting)
    plotImg = cv::Mat::zeros(600, 800, CV_8UC3) + cv::Scalar(255, 255, 255);

    // Initialize UKF parameters
    cv::Mat wm, wc;
    double gamma;
    setup_ukf(nx, wm, wc, gamma);

    while (time <= SIM_TIME) {
        time += dt;
        auto u = control_input();
        observation(xTrue, z, xDR, u);
        UKF_estimation(xEst, PEst, z, u, wm, wc, gamma, Q, R);

        // Store data history for plotting
        // Note: Conversion from cv::Mat to cv::Point2f for plotting
        hxEst.push_back(cv::Point2f(xEst.at<double>(0), xEst.at<double>(1)));
        hxDR.push_back(cv::Point2f(xDR.at<double>(0), xDR.at<double>(1)));
        hxTrue.push_back(cv::Point2f(xTrue.at<double>(0), xTrue.at<double>(1)));
        hz.push_back(cv::Point2f(z.at<double>(0), z.at<double>(1)));

        //plot_trajectories(plotImg, hxTrue, hxEst, hxDR);
        draw(plotImg, xEst, PEst);

        x = x + v * std::cos(yaw_rate * time) * dt * x_scale;
        y = y + v * std::sin(yaw_rate * time) * dt * y_scale;
        true_current = cv::Point(x, y);
        cv::line(plotImg, true_last, true_current, cv::Scalar(0, 0, 0), 2);
        true_last = true_current;
    }
}


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKF_UKF_test>());
    rclcpp::shutdown();
    return 0;
}