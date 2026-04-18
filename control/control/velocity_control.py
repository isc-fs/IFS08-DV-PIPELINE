import numpy as np
import rclpy
import rclpy.logging
from scipy.interpolate import UnivariateSpline

from control.utils import Derivative


class VelocityControl:
    """
    PID-based velocity controller with feedforward compensation for path curvature.

    This controller computes throttle/brake commands to track a target speed that is
    dynamically adjusted based on the upcoming path curvature and cross-track error.
    It uses a PID feedback loop combined with a feedforward term that estimates
    safe speeds through curves based on maximum lateral acceleration limits.
    """

    def __init__(
        self,
        K: float,
        Ki: float,
        Kd: float,
        Fg: float,
        max_speed: float = 10,
        max_normal_acceleration: float = 7,
        throttle_max: float = 0.4,
        brake_max: float = 0.4,
        smoothing_factor: float = 0.3,
    ) -> None:
        """
        Initialize the velocity controller with PID and feedforward gains.

        Args:
            K: Proportional gain for the PID controller
            Ki: Integral gain for the PID controller (equivalent to 1/Ti)
            Kd: Derivative gain for the PID controller (equivalent to Td)
            Fg: Feedforward gain for cross-track error compensation
            max_speed: Maximum allowable vehicle speed in m/s
            max_normal_acceleration: Maximum lateral acceleration in m/s²
            throttle_max: Maximum throttle command value (0.0 to 1.0)
            brake_max: Maximum brake command value (0.0 to 1.0)
            smoothing_factor: Alpha value for exponential moving average of target speed (0.0 to 1.0)
        """
        self.max_speed: float = max_speed
        self.max_normal_acceleration: float = max_normal_acceleration
        self.K: float = K  # Proportional gain
        self.Ki: float = Ki  # Integral gain
        self.Kd: float = Kd  # Derivative gain
        self.Fg: float = Fg  # Feedforward gain for cross-track error
        self.integral_error: float = 0.0  # Accumulated integral error for PID
        self.error_derivative: Derivative = Derivative()  # Derivative calculator
        self.throttle_max: float = throttle_max
        self.brake_max: float = brake_max
        self.target_speed_smoothing_factor: float = (
            smoothing_factor  # Smoothing factor for target speed
        )
        self.filtered_target_speed: float = 0.0  # Exponentially filtered target speed

    def get_feedforward_value(self, next_points: list[list[float]]) -> float:
        """
        Calculate the feedforward target speed based on upcoming path curvature.

        The target speed is computed using the relation: v = sqrt(r * a_max), where
        r is the path radius and a_max is the maximum lateral acceleration. This
        ensures the vehicle can negotiate curves without exceeding lateral g-limits.

        Args:
            next_points: List of [x, y] coordinates representing the upcoming path

        Returns:
            Target speed in m/s, clamped to max_speed
        """
        radius = self.get_radius(next_points)
        # Calculate safe speed for the curve: v = sqrt(r * a_max)
        # Saturate to max_speed to avoid exceeding vehicle limits
        return min(self.max_speed, np.sqrt(radius * self.max_normal_acceleration))

    def get_radius(self, next_points: list[list[float]]) -> float:
        """
        Estimate the minimum radius of curvature from a sequence of path points.

        Uses different methods depending on the number of available points:
        - < 3 points: Returns a large default radius
        - 3-4 points: Uses circumradius of triangles formed by consecutive points
        - >= 5 points: Uses parametric spline curve fitting for better accuracy

        Args:
            next_points: List of [x, y] coordinates representing the upcoming path

        Returns:
            Estimated minimum radius of curvature in meters
        """
        # Convert points to numpy arrays for vectorized operations
        next_points = [np.array(point) for point in next_points]

        if len(next_points) < 5:
            # Fallback to circumradius method for insufficient points
            if len(next_points) < 3:
                # Not enough points to estimate curvature, return large radius
                return self.max_speed**2 / (2 * self.max_normal_acceleration)
            else:
                # Calculate circumradius for triangles formed by consecutive triplets
                points_radius = []
                for i in range(min(3, len(next_points) - 2)):
                    # Triangle side lengths
                    a = np.linalg.norm(next_points[i] - next_points[i + 1])
                    b = np.linalg.norm(next_points[i + 1] - next_points[i + 2])
                    c = np.linalg.norm(next_points[i + 2] - next_points[i])
                    points_radius.append(self.calculate_circumradius(a, b, c))
                # Return the tightest (minimum) radius found
                return min(points_radius)
        else:
            # Use curve fitting method for better accuracy with sufficient points
            return self.estimate_radius_from_curve_fit(next_points)

    def calculate_circumradius(self, a: float, b: float, c: float) -> float:
        """
        Calculate the circumradius of a triangle given its three sides.

        The circumradius is the radius of the circle that passes through all three
        vertices of the triangle. It provides an approximation of path curvature.

        Formula: R = (abc) / (4K), where K is the triangle area from Heron's formula

        Args:
            a: Length of the first side of the triangle
            b: Length of the second side of the triangle
            c: Length of the third side of the triangle

        Returns:
            The circumradius of the triangle in meters (1000m if degenerate triangle)
        """
        # Calculate semi-perimeter
        s = (a + b + c) / 2

        # Calculate area using Heron's formula: K = sqrt(s(s-a)(s-b)(s-c))
        area = np.sqrt(abs(s * (s - a) * (s - b) * (s - c)))

        # Calculate circumradius: R = abc / (4K)
        radius = (a * b * c) / (4 * area) if area != 0 else 1000.0

        return radius

    def estimate_radius_from_curve_fit(self, points: list[np.ndarray]) -> float:
        """
        Estimate radius of curvature using parametric spline curve fitting.

        This method fits smooth splines to the x and y coordinates separately,
        then computes curvature at multiple points along the fitted curve. The
        minimum radius (maximum curvature) is returned to ensure conservative
        speed selection through the tightest part of the curve.

        Args:
            points: List of at least 5 points as numpy arrays [x, y]

        Returns:
            Estimated minimum radius of curvature in meters, clamped to [0.1, 100]
        """
        try:
            # Method 1: Parametric spline fitting
            spline_radius = self.parametric_spline_curvature(points)

            # Note: Previously used polynomial fitting as backup method
            # poly_radius = self.polynomial_curvature(points)
            # estimated_radius = min(spline_radius, poly_radius)

            # Clamp radius to reasonable physical bounds
            min_radius = 0.1  # Minimum radius of 0.1 meters (very tight turn)
            max_radius = 100.0  # Maximum radius of 100 meters (nearly straight)

            return max(min_radius, min(spline_radius, max_radius))

        except Exception as e:
            # Fallback to default large radius if curve fitting fails
            rclpy.logging.get_logger("velocity_control").debug(
                f"Curve fitting failed: {e}"
            )
            # Return a conservative large radius based on max speed and max acceleration
            return self.max_speed**2 / (2 * self.max_normal_acceleration)

    def parametric_spline_curvature(self, points: list[np.ndarray]) -> float:
        """
        Calculate curvature using parametric spline interpolation.

        This method parameterizes the curve by arc length and fits separate splines
        for x(t) and y(t). Curvature is computed using the formula:
        κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        where primes denote derivatives with respect to the parameter t.

        Args:
            points: List of points as numpy arrays [x, y]

        Returns:
            Minimum radius of curvature in meters (inverse of maximum curvature)
        """
        # Extract x and y coordinates from the points
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])

        # Create parameter t based on cumulative arc length (distance along path)
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            # Accumulate distance from previous point
            distances[i] = distances[i - 1] + np.linalg.norm(points[i] - points[i - 1])

        # Normalize parameter t to [0, 1] for spline fitting
        if distances[-1] > 0:
            t = distances / distances[-1]
        else:
            # All points are at the same location, return default large radius
            return self.max_speed**2 / (2 * self.max_normal_acceleration)

        # Fit cubic splines for x(t) and y(t) with adaptive smoothing
        # Smoothing factor is proportional to number of points to balance fit quality
        smoothing_factor = len(points) * 0.1
        spline_x = UnivariateSpline(t, x_coords, s=smoothing_factor, k=3)
        spline_y = UnivariateSpline(t, y_coords, s=smoothing_factor, k=3)

        # Calculate curvatures at multiple sample points along the curve
        curvatures = []
        # Evaluate at up to 10 points, avoiding endpoints (0.1 to 0.9)
        t_eval = np.linspace(0.1, 0.9, min(10, len(points)))

        for t_val in t_eval:
            # Compute first derivatives: dx/dt and dy/dt
            dx_dt = spline_x.derivative(1)(t_val)
            dy_dt = spline_y.derivative(1)(t_val)

            # Compute second derivatives: d²x/dt² and d²y/dt²
            d2x_dt2 = spline_x.derivative(2)(t_val)
            d2y_dt2 = spline_y.derivative(2)(t_val)

            # Apply curvature formula: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
            # This is the signed curvature for parametric curves
            numerator = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
            denominator = (dx_dt**2 + dy_dt**2) ** (3 / 2)

            # Only compute curvature if denominator is not near zero
            if denominator > 1e-8:
                curvature = numerator / denominator
                # Only store meaningful (non-negligible) curvatures
                if curvature > 1e-8:
                    curvatures.append(curvature)

        # If no valid curvatures were found, return default large radius
        if not curvatures:
            return self.max_speed**2 / (2 * self.max_normal_acceleration)

        # Find the maximum curvature (tightest part of the curve)
        # and return its corresponding radius (inverse of curvature)
        max_curvature = max(curvatures)
        return 1.0 / max_curvature

    def get_control_value(
        self,
        velocity_measurement: float,
        next_points: list[list[float]],
        crosstrack_error: float,
    ) -> tuple[float, float]:
        """
        Compute the throttle/brake command using PID control with feedforward.

        The controller computes a target speed based on upcoming path curvature,
        reduces it based on cross-track error, applies exponential smoothing,
        and then uses a PID controller to track the smoothed target speed.

        The integral term is only updated when the command is not saturated
        (anti-windup mechanism).

        Args:
            velocity_measurement: Current longitudinal velocity in m/s
            next_points: List of [x, y] coordinates for the upcoming path
            crosstrack_error: Lateral deviation from the reference path in meters

        Returns:
            A tuple containing:
                - command_value: Throttle (positive) or brake (negative) command
                - speed_margin: Difference between max_speed and feedforward target
        """
        # Calculate base target speed from path curvature (feedforward term)
        feedforward_value = self.get_feedforward_value(next_points)

        # Reduce target speed if cross-track error is significant (> 0.2m)
        # Uses quadratic penalty to aggressively slow down when off-track
        crosstrack_term = (
            self.Fg * abs(crosstrack_error**2) if abs(crosstrack_error) > 0.2 else 0.0
        )

        # Compute raw target speed and ensure it's non-negative
        raw_target_speed = feedforward_value - crosstrack_term
        raw_target_speed = max(0.0, raw_target_speed)

        # Apply exponential moving average (EMA) to smooth target speed changes
        # This prevents abrupt speed commands that could destabilize the vehicle
        alpha = self.target_speed_smoothing_factor
        self.filtered_target_speed = (alpha * raw_target_speed) + (
            1 - alpha
        ) * self.filtered_target_speed

        # Use the smoothed target speed for the PID controller
        target_speed = self.filtered_target_speed

        # Compute tracking error
        error = target_speed - velocity_measurement

        # Calculate derivative of error for D term
        derivative = self.error_derivative.cal(error)

        # Compute PID control law: u = K * (e + Kd*de/dt + Ki*∫e dt)
        unsat_command_value = self.K * (
            error + self.Kd * derivative + self.Ki * self.integral_error
        )

        # Saturate command to actuator limits
        command_value = np.clip(unsat_command_value, -self.brake_max, self.throttle_max)

        # Anti-windup: only integrate error when command is not saturated
        # This prevents integral windup during saturation
        tolerance = 1e-5
        if abs(command_value - unsat_command_value) < tolerance:
            self.integral_error += error

        # Return command and speed margin for diagnostics
        return command_value, self.max_speed - feedforward_value
