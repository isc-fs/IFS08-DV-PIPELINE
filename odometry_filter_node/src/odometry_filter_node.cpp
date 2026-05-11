// Copyright 2026 IFSSIM contributors.
//
// odometry_filter_node — rclcpp wrapper around odometry_filter::OdometryFilter.
//
// Subscribes:
//   /imu                (sensor_msgs/Imu, BEST_EFFORT, deep queue —
//                        BMI088 native ~400 Hz; deep queue absorbs jitter
//                        when the publish timer is mid-tick).
//   /motor_rpm          (std_msgs/Float32, BEST_EFFORT, depth 10 — ~80 Hz)
//   /steering_angle     (std_msgs/Float32, BEST_EFFORT, depth 10 — ~100 Hz)
//   /brake_pressure     (std_msgs/Float32, BEST_EFFORT, depth 10 — ~100 Hz)
//
// Publishes:
//   /odom               (nav_msgs/Odometry @ 100 Hz, lifecycle pub)
//   /odom_diag/yaw_residual_rad_s  (std_msgs/Float32 @ 100 Hz)
//   /odom_diag/slip_flag           (std_msgs/Bool @ 100 Hz)
//   /odom_diag/effective_alpha_vx  (std_msgs/Float32 @ 100 Hz)
//
// Broadcasts:
//   odom → base_link TF @ 100 Hz (alongside the Odometry message —
//   identical pose; the TF copy is for tf2 consumers).
//
// Lifecycle layout — mirrors sim_supervisor_node.py's filter section:
//   on_configure:  create pubs (lifecycle) + TF broadcaster, instantiate filter.
//   on_activate:   reset filter, subscribe, start 100 Hz publish timer.
//                  pubs flip to "emitting" state via super().on_activate().
//   on_deactivate: drop subs + timer.
//   on_cleanup:    drop pubs, TF broadcaster, filter.
//
// Coexistence with sim_supervisor: until Phase 3 strips the Python
// filter, sim_supervisor still publishes /odom too. They publish to
// the SAME topic — last-writer-wins is fine for testing, just be
// aware. Phase 3 adds a `use_external_odometry_filter` param to
// sim_supervisor and flips this node on by default.

#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "odometry_filter/odometry_filter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>

#include <tf2_ros/transform_broadcaster.h>

using rclcpp_lifecycle::LifecycleNode;
using rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface;
using CallbackReturn = LifecycleNodeInterface::CallbackReturn;

namespace odometry_filter_node {

class OdometryFilterNode : public LifecycleNode {
 public:
  explicit OdometryFilterNode(const rclcpp::NodeOptions & options)
  : LifecycleNode("odometry_filter_node", options) {
    declare_parameter<double>("publish_hz", 100.0);
    declare_parameter<std::string>("odom_frame", "odom");
    declare_parameter<std::string>("base_frame", "base_link");
  }

  // ------------------------------------------------------------------
  // Lifecycle transitions
  // ------------------------------------------------------------------
  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override {
    RCLCPP_INFO(get_logger(),
      "on_configure: lifecycle pubs + TF broadcaster + filter");

    publish_hz_ = get_parameter("publish_hz").as_double();
    odom_frame_ = get_parameter("odom_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom", 50);
    yaw_residual_pub_ = create_publisher<std_msgs::msg::Float32>(
      "/odom_diag/yaw_residual_rad_s", 10);
    slip_flag_pub_ = create_publisher<std_msgs::msg::Bool>(
      "/odom_diag/slip_flag", 10);
    effective_alpha_pub_ = create_publisher<std_msgs::msg::Float32>(
      "/odom_diag/effective_alpha_vx", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    filter_ = std::make_unique<odometry_filter::OdometryFilter>();
    first_publish_logged_ = false;

    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override {
    RCLCPP_INFO(get_logger(),
      "on_activate: subs + %.0f Hz publish timer", publish_hz_);

    // Reset the filter so a deactivate→activate cycle starts a fresh
    // stationary calibration — matches sim_supervisor_node behavior.
    filter_->reset();
    first_publish_logged_ = false;

    // IMU — BEST_EFFORT, deep queue (2000). The bridge publishes at
    // ~400 Hz; we want to absorb backlog whenever the publish timer
    // happens to be mid-tick. Matches sim_supervisor_node's QoS.
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
    sub_brake_ = create_subscription<std_msgs::msg::Float32>(
      "/brake_pressure", rpm_qos,
      std::bind(&OdometryFilterNode::on_brake, this, std::placeholders::_1));

    const auto period = std::chrono::duration<double>(1.0 / publish_hz_);
    publish_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&OdometryFilterNode::publish_odom, this));

    return LifecycleNode::on_activate(state);
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override {
    RCLCPP_INFO(get_logger(), "on_deactivate: dropping subs + timer");
    publish_timer_.reset();
    sub_imu_.reset();
    sub_rpm_.reset();
    sub_steering_.reset();
    sub_brake_.reset();
    return LifecycleNode::on_deactivate(state);
  }

  CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override {
    RCLCPP_INFO(get_logger(), "on_cleanup: dropping pubs + filter");
    publish_timer_.reset();
    sub_imu_.reset();
    sub_rpm_.reset();
    sub_steering_.reset();
    sub_brake_.reset();
    odom_pub_.reset();
    yaw_residual_pub_.reset();
    slip_flag_pub_.reset();
    effective_alpha_pub_.reset();
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
    // Wall-clock timestamp — the bridge publishes Float32 with no
    // header.stamp on /motor_rpm. Used for staleness inside the filter.
    const double t = now().seconds();
    filter_->push_rpm(t, static_cast<double>(msg->data));
  }

  void on_steering(const std_msgs::msg::Float32::SharedPtr msg) {
    if (!filter_) {return;}
    filter_->push_steering(now().seconds(), static_cast<double>(msg->data));
  }

  void on_brake(const std_msgs::msg::Float32::SharedPtr msg) {
    if (!filter_) {return;}
    filter_->push_brake(now().seconds(), static_cast<double>(msg->data));
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
        "/odom first publish — IMU+RPM filter calibrated");
      first_publish_logged_ = true;
    }

    // 2D yaw → unit quaternion (axis-z) used by both the Odometry
    // message and the TF broadcast.
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

    std_msgs::msg::Float32 alpha;
    alpha.data = static_cast<float>(diag.effective_alpha_vx);
    effective_alpha_pub_->publish(alpha);
  }

  // ------------------------------------------------------------------
  // Members
  // ------------------------------------------------------------------
  std::unique_ptr<odometry_filter::OdometryFilter> filter_;

  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float32>::SharedPtr yaw_residual_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>::SharedPtr slip_flag_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float32>::SharedPtr effective_alpha_pub_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_rpm_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_steering_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_brake_;

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
