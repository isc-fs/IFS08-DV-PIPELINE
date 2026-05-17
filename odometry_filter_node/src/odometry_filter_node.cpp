// Copyright 2026 IFSSIM contributors.
//
// odometry_filter_node — rclcpp wrapper around odometry_filter's 9-state EKF.
//
// Subscribes:
//   /imu                (sensor_msgs/Imu, BEST_EFFORT, deep queue —
//                        BMI088 native ~400 Hz; deep queue absorbs jitter
//                        when the publish timer is mid-tick).
//   /motor_rpm          (std_msgs/Float32, BEST_EFFORT, depth 10 — ~80 Hz)
//   /steering_angle     (std_msgs/Float32, BEST_EFFORT, depth 10 — ~100 Hz)
//
// Publishes:
//   /odom                          (nav_msgs/Odometry @ 100 Hz, lifecycle pub)
//   /odom_diag/yaw_residual_rad_s  (std_msgs/Float32 @ 100 Hz)
//   /odom_diag/slip_flag           (std_msgs/Bool    @ 100 Hz)
//
// Broadcasts:
//   odom → base_link TF @ 100 Hz (alongside the Odometry message —
//   identical pose; the TF copy is for tf2 consumers).
//
// Lifecycle layout (BaseLifecycleNode pattern from PR #518):
//   on_configure_impl: create pubs (lifecycle) + TF broadcaster,
//                      construct filter from ROS params.
//   on_activate:       reset filter, subscribe, start 100 Hz publish
//                      timer; super().on_activate() flips lifecycle
//                      pubs to "emitting".
//   on_deactivate:     drop subs + timer.
//   on_cleanup_impl:   drop pubs, TF broadcaster, filter.

#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "odometry_filter/odometry_filter.hpp"

#include <node_base_cpp/base_lifecycle_node.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>

#include <tf2_ros/transform_broadcaster.h>

using rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface;
using CallbackReturn = LifecycleNodeInterface::CallbackReturn;

namespace odometry_filter_node {

namespace {
// Sentinel for nav_msgs/Odometry covariance entries we don't estimate
// (z, roll, pitch, vz, ωx, ωy). Following the REP-103 convention,
// a large value signals "this DoF is not observed."
constexpr double kCovUnknown = 1.0e6;
}  // namespace

class OdometryFilterNode : public node_base_cpp::BaseLifecycleNode {
 public:
  explicit OdometryFilterNode(const rclcpp::NodeOptions & options)
  : node_base_cpp::BaseLifecycleNode("odometry_filter_node", options) {
    // Topology / publish-loop knobs.
    declare_parameter<double>("publish_hz", 100.0);
    declare_parameter<std::string>("odom_frame", "odom");
    declare_parameter<std::string>("base_frame", "base_link");

    // EKF tuning knobs, exposed via ~/ekf.* for YAML overrides.
    declare_parameter<double>("ekf.wheelbase_m",        odometry_filter::kWheelbaseM);
    declare_parameter<double>("ekf.rpm_to_ms",          odometry_filter::kRpmToMs);
    declare_parameter<double>("ekf.calibration_seconds",odometry_filter::kCalibrationSeconds);
    declare_parameter<double>("ekf.sigma_ax",           0.05);
    declare_parameter<double>("ekf.sigma_ay",           0.05);
    declare_parameter<double>("ekf.sigma_gz",           0.01);
    declare_parameter<double>("ekf.sigma_ba_walk",      1.0e-4);
    declare_parameter<double>("ekf.sigma_bg_walk",      1.0e-5);
    declare_parameter<double>("ekf.sigma_rpm",          0.02);
    declare_parameter<double>("ekf.sigma_steer",        0.30);
    declare_parameter<double>("ekf.sigma_vy_nhc",       0.10);
    declare_parameter<double>("ekf.slip_yaw_residual_threshold",
                              odometry_filter::kSlipYawResidualThreshold);
    declare_parameter<double>("ekf.min_vx_for_steering_correct",
                              odometry_filter::kMinVxForSteeringCorrect);
  }

  // ------------------------------------------------------------------
  // Lifecycle transitions
  // ------------------------------------------------------------------
  CallbackReturn on_configure_impl(const rclcpp_lifecycle::State &) override {
    RCLCPP_INFO(get_logger(),
      "on_configure: lifecycle pubs + TF broadcaster + EKF (Coriolis-correct)");

    publish_hz_ = get_parameter("publish_hz").as_double();
    odom_frame_ = get_parameter("odom_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom", 50);
    yaw_residual_pub_ = create_publisher<std_msgs::msg::Float32>(
      "/odom_diag/yaw_residual_rad_s", 10);
    slip_flag_pub_ = create_publisher<std_msgs::msg::Bool>(
      "/odom_diag/slip_flag", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    odometry_filter::EkfParams p;
    p.wheelbase_m              = get_parameter("ekf.wheelbase_m").as_double();
    p.rpm_to_ms                = get_parameter("ekf.rpm_to_ms").as_double();
    p.calibration_seconds      = get_parameter("ekf.calibration_seconds").as_double();
    p.sigma_ax                 = get_parameter("ekf.sigma_ax").as_double();
    p.sigma_ay                 = get_parameter("ekf.sigma_ay").as_double();
    p.sigma_gz                 = get_parameter("ekf.sigma_gz").as_double();
    p.sigma_ba_walk            = get_parameter("ekf.sigma_ba_walk").as_double();
    p.sigma_bg_walk            = get_parameter("ekf.sigma_bg_walk").as_double();
    p.sigma_rpm                = get_parameter("ekf.sigma_rpm").as_double();
    p.sigma_steer              = get_parameter("ekf.sigma_steer").as_double();
    p.sigma_vy_nhc             = get_parameter("ekf.sigma_vy_nhc").as_double();
    p.slip_yaw_residual_threshold =
      get_parameter("ekf.slip_yaw_residual_threshold").as_double();
    p.min_vx_for_steering_correct =
      get_parameter("ekf.min_vx_for_steering_correct").as_double();

    filter_ = std::make_unique<odometry_filter::OdometryFilter>(p);
    first_publish_logged_ = false;

    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override {
    RCLCPP_INFO(get_logger(),
      "on_activate: subs + %.0f Hz publish timer", publish_hz_);

    // Reset the filter so a deactivate→activate cycle starts a fresh
    // stationary calibration window.
    filter_->reset();
    first_publish_logged_ = false;

    // IMU — BEST_EFFORT, deep queue (2000). The bridge publishes at
    // ~400 Hz; we want to absorb backlog whenever the publish timer
    // happens to be mid-tick.
    auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(2000))
                     .best_effort()
                     .durability_volatile();
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu", imu_qos,
      std::bind(&OdometryFilterNode::on_imu, this, std::placeholders::_1));

    auto rpm_qos = rclcpp::QoS(rclcpp::KeepLast(10))
                     .best_effort()
                     .durability_volatile();
    sub_rpm_ = create_subscription<std_msgs::msg::Float32>(
      "/motor_rpm", rpm_qos,
      std::bind(&OdometryFilterNode::on_rpm, this, std::placeholders::_1));
    sub_steering_ = create_subscription<std_msgs::msg::Float32>(
      "/steering_angle", rpm_qos,
      std::bind(&OdometryFilterNode::on_steering, this, std::placeholders::_1));

    const auto period = std::chrono::duration<double>(1.0 / publish_hz_);
    publish_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&OdometryFilterNode::publish_odom, this));

    return BaseLifecycleNode::on_activate(state);
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override {
    RCLCPP_INFO(get_logger(), "on_deactivate: dropping subs + timer");
    publish_timer_.reset();
    sub_imu_.reset();
    sub_rpm_.reset();
    sub_steering_.reset();
    return BaseLifecycleNode::on_deactivate(state);
  }

  CallbackReturn on_cleanup_impl(const rclcpp_lifecycle::State &) override {
    RCLCPP_INFO(get_logger(), "on_cleanup: dropping pubs + filter");
    publish_timer_.reset();
    sub_imu_.reset();
    sub_rpm_.reset();
    sub_steering_.reset();
    odom_pub_.reset();
    yaw_residual_pub_.reset();
    slip_flag_pub_.reset();
    tf_broadcaster_.reset();
    filter_.reset();
    return CallbackReturn::SUCCESS;
  }

 private:
  // ------------------------------------------------------------------
  // Callbacks
  // ------------------------------------------------------------------
  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg) {
    if (!filter_) {return;}
    const double t = static_cast<double>(msg->header.stamp.sec) +
                     static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    const Eigen::Vector3d accel(
      msg->linear_acceleration.x,
      msg->linear_acceleration.y,
      msg->linear_acceleration.z);
    const Eigen::Vector3d gyro(
      msg->angular_velocity.x,
      msg->angular_velocity.y,
      msg->angular_velocity.z);
    filter_->push_imu(t, accel, gyro);
  }

  void on_rpm(const std_msgs::msg::Float32::SharedPtr msg) {
    if (!filter_) {return;}
    filter_->push_rpm(now().seconds(), static_cast<double>(msg->data));
  }

  void on_steering(const std_msgs::msg::Float32::SharedPtr msg) {
    if (!filter_) {return;}
    filter_->push_steering(now().seconds(), static_cast<double>(msg->data));
  }

  // ------------------------------------------------------------------
  // Publish timer
  // ------------------------------------------------------------------
  void publish_odom() {
    if (!filter_ || !filter_->is_calibrated()) {return;}

    const auto & s = filter_->state();
    const auto stamp = this->now();

    if (!first_publish_logged_) {
      RCLCPP_INFO(get_logger(),
        "/odom first publish — EKF calibrated");
      first_publish_logged_ = true;
    }

    // 2D yaw → unit quaternion (axis-z).
    const double half = 0.5 * s.yaw;
    const double qw = std::cos(half);
    const double qz = std::sin(half);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = s.x;
    odom.pose.pose.position.y = s.y;
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation.w = qw;
    odom.pose.pose.orientation.x = 0.0;
    odom.pose.pose.orientation.y = 0.0;
    odom.pose.pose.orientation.z = qz;
    odom.twist.twist.linear.x = s.vx;
    odom.twist.twist.linear.y = s.vy;
    odom.twist.twist.linear.z = 0.0;
    odom.twist.twist.angular.z = s.yaw_rate;

    // Populate covariance from the EKF P matrix. nav_msgs/Odometry
    // wants 6×6 row-major for (x,y,z,roll,pitch,yaw) on pose and
    // (vx,vy,vz,ωx,ωy,ωz) on twist. We estimate the planar slice
    // (x,y,yaw / vx,vy,ω); z/roll/pitch/vz/ωx/ωy get kCovUnknown.
    const auto P = filter_->covariance();
    // Pose: x, y, z, roll, pitch, yaw.
    odom.pose.covariance.fill(0.0);
    odom.pose.covariance[6 * 0 + 0] = P(odometry_filter::X, odometry_filter::X);
    odom.pose.covariance[6 * 0 + 1] = P(odometry_filter::X, odometry_filter::Y);
    odom.pose.covariance[6 * 0 + 5] = P(odometry_filter::X, odometry_filter::THETA);
    odom.pose.covariance[6 * 1 + 0] = P(odometry_filter::Y, odometry_filter::X);
    odom.pose.covariance[6 * 1 + 1] = P(odometry_filter::Y, odometry_filter::Y);
    odom.pose.covariance[6 * 1 + 5] = P(odometry_filter::Y, odometry_filter::THETA);
    odom.pose.covariance[6 * 2 + 2] = kCovUnknown;  // z
    odom.pose.covariance[6 * 3 + 3] = kCovUnknown;  // roll
    odom.pose.covariance[6 * 4 + 4] = kCovUnknown;  // pitch
    odom.pose.covariance[6 * 5 + 0] = P(odometry_filter::THETA, odometry_filter::X);
    odom.pose.covariance[6 * 5 + 1] = P(odometry_filter::THETA, odometry_filter::Y);
    odom.pose.covariance[6 * 5 + 5] = P(odometry_filter::THETA, odometry_filter::THETA);
    // Twist: vx, vy, vz, ωx, ωy, ωz.
    odom.twist.covariance.fill(0.0);
    odom.twist.covariance[6 * 0 + 0] = P(odometry_filter::VX, odometry_filter::VX);
    odom.twist.covariance[6 * 0 + 1] = P(odometry_filter::VX, odometry_filter::VY);
    odom.twist.covariance[6 * 0 + 5] = P(odometry_filter::VX, odometry_filter::OMEGA);
    odom.twist.covariance[6 * 1 + 0] = P(odometry_filter::VY, odometry_filter::VX);
    odom.twist.covariance[6 * 1 + 1] = P(odometry_filter::VY, odometry_filter::VY);
    odom.twist.covariance[6 * 1 + 5] = P(odometry_filter::VY, odometry_filter::OMEGA);
    odom.twist.covariance[6 * 2 + 2] = kCovUnknown;  // vz
    odom.twist.covariance[6 * 3 + 3] = kCovUnknown;  // ωx
    odom.twist.covariance[6 * 4 + 4] = kCovUnknown;  // ωy
    odom.twist.covariance[6 * 5 + 0] = P(odometry_filter::OMEGA, odometry_filter::VX);
    odom.twist.covariance[6 * 5 + 1] = P(odometry_filter::OMEGA, odometry_filter::VY);
    odom.twist.covariance[6 * 5 + 5] = P(odometry_filter::OMEGA, odometry_filter::OMEGA);

    odom_pub_->publish(odom);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = odom_frame_;
    tf_msg.child_frame_id = base_frame_;
    tf_msg.transform.translation.x = s.x;
    tf_msg.transform.translation.y = s.y;
    tf_msg.transform.translation.z = 0.0;
    tf_msg.transform.rotation.w = qw;
    tf_msg.transform.rotation.x = 0.0;
    tf_msg.transform.rotation.y = 0.0;
    tf_msg.transform.rotation.z = qz;
    tf_broadcaster_->sendTransform(tf_msg);

    // Diagnostics — cheap, fire every tick alongside /odom.
    const auto & diag = filter_->diagnostics();
    std_msgs::msg::Float32 yaw_res;
    yaw_res.data = static_cast<float>(diag.yaw_residual_rad_s);
    yaw_residual_pub_->publish(yaw_res);

    std_msgs::msg::Bool slip;
    slip.data = diag.slip_flag;
    slip_flag_pub_->publish(slip);
  }

  // ------------------------------------------------------------------
  // Members
  // ------------------------------------------------------------------
  std::unique_ptr<odometry_filter::OdometryFilter> filter_;

  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float32>::SharedPtr yaw_residual_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>::SharedPtr slip_flag_pub_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_rpm_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_steering_;

  rclcpp::TimerBase::SharedPtr publish_timer_;

  double publish_hz_{100.0};
  std::string odom_frame_{"odom"};
  std::string base_frame_{"base_link"};
  bool first_publish_logged_{false};
};

}  // namespace odometry_filter_node


int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<odometry_filter_node::OdometryFilterNode>(options);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
