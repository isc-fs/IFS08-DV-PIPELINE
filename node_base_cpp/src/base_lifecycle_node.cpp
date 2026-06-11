#include "node_base_cpp/base_lifecycle_node.hpp"

namespace node_base_cpp {

BaseLifecycleNode::BaseLifecycleNode(
  const std::string & node_name, const rclcpp::NodeOptions & options)
: LifecycleNode(node_name, options)
{
  setup_srv_ = create_service<dv_msgs::srv::Setup>(
    "~/setup",
    std::bind(
      &BaseLifecycleNode::handle_setup, this,
      std::placeholders::_1, std::placeholders::_2));
}

void BaseLifecycleNode::handle_setup(
  const std::shared_ptr<dv_msgs::srv::Setup::Request> request,
  std::shared_ptr<dv_msgs::srv::Setup::Response> response)
{
  mode_name_ = request->mode_name;
  behavior_ = request->behavior;
  response->success = true;
  response->message =
    "Ready: mode=" + mode_name_ + " behavior=" + behavior_;
  RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
}

BaseLifecycleNode::CallbackReturn BaseLifecycleNode::on_configure(
  const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(
    get_logger(), "Configured | mode: %s | behavior: %s",
    mode_name_.c_str(), behavior_.c_str());
  return on_configure_impl(state);
}

BaseLifecycleNode::CallbackReturn BaseLifecycleNode::on_configure_impl(
  const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

BaseLifecycleNode::CallbackReturn BaseLifecycleNode::on_activate(
  const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Activated in mode: %s", mode_name_.c_str());
  return LifecycleNode::on_activate(state);
}

BaseLifecycleNode::CallbackReturn BaseLifecycleNode::on_deactivate(
  const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Deactivated");
  return LifecycleNode::on_deactivate(state);
}

BaseLifecycleNode::CallbackReturn BaseLifecycleNode::on_cleanup(
  const rclcpp_lifecycle::State & state)
{
  const auto ret = on_cleanup_impl(state);
  if (ret != CallbackReturn::SUCCESS) {
    return ret;
  }
  mode_name_.clear();
  behavior_ = "default";
  RCLCPP_INFO(get_logger(), "Cleaned up");
  return LifecycleNode::on_cleanup(state);
}

BaseLifecycleNode::CallbackReturn BaseLifecycleNode::on_cleanup_impl(
  const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

BaseLifecycleNode::CallbackReturn BaseLifecycleNode::on_shutdown(
  const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(get_logger(), "Shutdown");
  return LifecycleNode::on_shutdown(state);
}

}  // namespace node_base_cpp
