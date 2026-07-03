#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <dv_msgs/srv/setup.hpp>

namespace node_base_cpp {

/// Lifecycle node exposing ~/setup for mode_manager (mirrors Python node_base).
class BaseLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
 public:
  explicit BaseLifecycleNode(
    const std::string & node_name,
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  const std::string & mode_name() const { return mode_name_; }
  const std::string & behavior() const { return behavior_; }

 protected:
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

  /// Called after setup fields are stored; override for node-specific logic.
  virtual CallbackReturn on_configure_impl(const rclcpp_lifecycle::State & state);
  virtual CallbackReturn on_cleanup_impl(const rclcpp_lifecycle::State & state);

  std::string mode_name_;
  std::string behavior_{"default"};

 private:
  void handle_setup(
    const std::shared_ptr<dv_msgs::srv::Setup::Request> request,
    std::shared_ptr<dv_msgs::srv::Setup::Response> response);

  rclcpp::Service<dv_msgs::srv::Setup>::SharedPtr setup_srv_;
};

}  // namespace node_base_cpp
