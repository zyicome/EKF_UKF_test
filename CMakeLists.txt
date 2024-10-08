cmake_minimum_required(VERSION 3.8)
project(EKF_UKF_test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/EKF_UKF_test)

add_executable(EKF_UKF_test src/ekf_ukf_test.cpp)
ament_target_dependencies(EKF_UKF_test rclcpp OpenCV)

add_executable(rm_ekf_test src/rm_ekf_test.cpp)
ament_target_dependencies(rm_ekf_test rclcpp OpenCV)

install(TARGETS
  EKF_UKF_test
  rm_ekf_test
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
