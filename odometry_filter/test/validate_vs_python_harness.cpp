// Tiny CLI harness for validate_vs_python.py — reads event lines from
// stdin in the same schema the Python driver writes:
//
//   <t> imu <ax> <ay> <az> <gx> <gy> <gz>
//   <t> rpm <value>
//   <t> steer <angle_rad>
//   <t> brake <brake>
//
// Drains stdin, replays the events through OdometryFilter, then writes
// one line to stdout:
//
//   <x> <y> <yaw> <vx> <vy> <yaw_rate>
//
// validate_vs_python.py compares this against the Python reference
// run with the same input. Any divergence > 1e-9 fails the check.
// Stays in the test/ folder because it has no production purpose —
// it's the equivalence-with-Python harness, full stop.

#include <iostream>
#include <sstream>
#include <string>

#include "odometry_filter/odometry_filter.hpp"

using odometry_filter::OdometryFilter;

int main() {
  OdometryFilter f;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {continue;}
    std::istringstream iss(line);
    double t;
    std::string kind;
    iss >> t >> kind;
    if (kind == "imu") {
      double ax, ay, az, gx, gy, gz;
      iss >> ax >> ay >> az >> gx >> gy >> gz;
      f.push_imu(t, Eigen::Vector3d(ax, ay, az), Eigen::Vector3d(gx, gy, gz));
    } else if (kind == "rpm") {
      double rpm;
      iss >> rpm;
      f.push_rpm(t, rpm);
    } else if (kind == "steer") {
      double angle;
      iss >> angle;
      f.push_steering(t, angle);
    } else if (kind == "brake") {
      double brake;
      iss >> brake;
      f.push_brake(t, brake);
    }
  }
  const auto & s = f.state();
  std::cout.precision(17);
  std::cout << s.x << " " << s.y << " " << s.yaw << " "
            << s.vx << " " << s.vy << " " << s.yaw_rate << "\n";
  return 0;
}
