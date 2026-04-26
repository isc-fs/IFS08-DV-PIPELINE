"""
https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf
https://github.com/winstxnhdw/FullStanleyController.git
"""

from math import atan2, cos, sin

import numpy as np

from control.utils import wrap_to_pi


class StanleyController:

    def __init__(
        self,
        control_gain=2.5,
        softening_gain=6.0,
        yaw_rate_gain=0.0,
        steering_damp_gain=0.0,
        max_steer=np.deg2rad(25),
        wheelbase=1.55,  # FSG T 8 minimum is 1525 mm; real cars ~1.5–1.7 m
        path_x=None,
        path_y=None,
        path_yaw=None,
    ):
        # NOTE on `steering_damp_gain` semantics: the formula in
        # `stanley_control` below is
        #   output = desired - k_damp * (desired - prev)
        #         = (1 - k_damp) * desired + k_damp * prev
        # so k_damp = 0 → use desired (no damping), k_damp = 1.0 → output = prev
        # (controller fully overridden by its own previous output, locking at 0
        # since `prev` is fed back). Anything between is an exponential blend.
        # Counter-intuitive name — "more damp" gives less controller authority,
        # not "more smoothing." Keep ≤ 0.5 if you ever turn this on.
        """
        Stanley Controller

        At initialisation
        :param control_gain:                (float) time constant [1/s]
        :param softening_gain:              (float) softening gain [m/s]
        :param yaw_rate_gain:               (float) yaw rate gain [rad]
        :param steering_damp_gain:          (float) steering damp gain
        :param max_steer:                   (float) vehicle's steering limits [rad]
        :param wheelbase:                   (float) vehicle's wheelbase [m]
        :param path_x:                      (numpy.ndarray) list of x-coordinates along the path
        :param path_y:                      (numpy.ndarray) list of y-coordinates along the path
        :param path_yaw:                    (numpy.ndarray) list of discrete yaw values along the path
        :param dt:                          (float) discrete time period [s]

        At every time step
        :param x:                           (float) vehicle's x-coordinate [m]
        :param y:                           (float) vehicle's y-coordinate [m]
        :param yaw:                         (float) vehicle's heading [rad]
        :param target_velocity:             (float) vehicle's velocity [m/s]
        :param steering_angle:              (float) vehicle's steering angle [rad]

        :return limited_steering_angle:     (float) steering angle after imposing steering limits [rad]
        :return target_index:               (int) closest path index
        :return crosstrack_error:           (float) distance from closest path index [m]
        """

        self.k = control_gain
        self.k_soft = softening_gain
        self.k_yaw_rate = yaw_rate_gain
        self.k_damp_steer = steering_damp_gain
        self.max_steer = max_steer
        self.wheelbase = wheelbase

    def set_path(self, path_x, path_y, path_yaw):
        self.px = path_x
        self.py = path_y
        self.pyaw = path_yaw

    def find_target_path_id(self, x, y, yaw):

        # Calculate position of the front axle
        fx = x + self.wheelbase * cos(yaw)
        fy = y + self.wheelbase * sin(yaw)

        dx = fx - self.px  # Find the x-axis of the front axle relative to the path
        dy = fy - self.py  # Find the y-axis of the front axle relative to the path

        d = np.hypot(dx, dy)  # Find the distance from the front axle to the path
        target_index = np.argmin(d)  # Find the shortest distance in the array

        return target_index, dx[target_index], dy[target_index], d[target_index]

    def calculate_yaw_term(self, target_index, yaw):

        yaw_error = wrap_to_pi(self.pyaw[target_index] - yaw)

        return yaw_error

    def calculate_crosstrack_term(self, target_velocity, yaw, dx, dy, absolute_error):

        front_axle_vector = np.array([sin(yaw), -cos(yaw)])
        nearest_path_vector = np.array([dx, dy])
        crosstrack_error = (
            np.sign(nearest_path_vector @ front_axle_vector) * absolute_error
        )

        crosstrack_steering_error = atan2(
            (self.k * crosstrack_error), (self.k_soft + target_velocity)
        )

        return crosstrack_steering_error, crosstrack_error

    def calculate_yaw_rate_term(self, target_velocity, steering_angle):

        yaw_rate_error = (
            self.k_yaw_rate * (-target_velocity * sin(steering_angle)) / self.wheelbase
        )

        return yaw_rate_error

    def calculate_steering_delay_term(
        self, computed_steering_angle, previous_steering_angle
    ):

        steering_delay_error = self.k_damp_steer * (
            computed_steering_angle - previous_steering_angle
        )

        return steering_delay_error

    def stanley_control(self, x, y, yaw, target_velocity, steering_angle=0):

        target_index, dx, dy, absolute_error = self.find_target_path_id(x, y, yaw)
        yaw_error = self.calculate_yaw_term(target_index, yaw)
        crosstrack_steering_error, crosstrack_error = self.calculate_crosstrack_term(
            target_velocity, yaw, dx, dy, absolute_error
        )
        yaw_rate_damping = self.calculate_yaw_rate_term(target_velocity, steering_angle)

        desired_steering_angle = (
            yaw_error + crosstrack_steering_error + yaw_rate_damping
        )

        # Constrains steering angle to the vehicle limits
        desired_steering_angle -= self.calculate_steering_delay_term(
            desired_steering_angle, steering_angle
        )
        limited_steering_angle = np.clip(
            desired_steering_angle, -self.max_steer, self.max_steer
        )

        return limited_steering_angle, target_index, crosstrack_error

    def stanley_control_at_index(self, x, y, yaw, target_velocity, target_index,
                                 steering_angle=0, preview_index=None):
        # Hybrid Pure-Pursuit + Stanley:
        #
        #   • Cross-track (lateral) term — classic Stanley
        #         δ_xte = atan2(k · e_xte, k_soft + v)
        #     using the NEAREST path point at target_index. Honest "how
        #     off-path am I now" feedback.
        #
        #   • Heading (longitudinal) term — Pure-Pursuit geometric law
        #         δ_pp  = atan2(2 · L · sin α, L_d_actual)
        #     where α is the bearing from the FRONT AXLE to the preview
        #     point at preview_index, and L_d_actual is the chord length
        #     to that point. δ_pp is the steer angle that would carry
        #     the chassis through the preview point on a circular arc,
        #     so it tracks chassis kinematics correctly. The previous
        #     formulation (Preview-Stanley) injected `pyaw[preview] - yaw`
        #     directly as a steer command and over-steered: at v = 6.9
        #     m/s on a 17 m-radius corner it commanded 13° where the
        #     geometry only requires arctan(L/R) ≈ 5.4°. That over-shot
        #     into the curve interior on every entry.
        #
        # The arc-length anchor on target_index (in control.py) keeps
        # this stable across path replans; without it the natural argmin
        # would leap several metres at a curve where the planner's path
        # changes shape between solves (DIAG: target index jumped 6.9 m
        # in 200 ms before the anchor was added).
        if preview_index is None:
            preview_index = target_index

        fx = x + self.wheelbase * cos(yaw)
        fy = y + self.wheelbase * sin(yaw)

        # Cross-track at nearest target_index
        dx = fx - self.px[target_index]
        dy = fy - self.py[target_index]
        absolute_error = float(np.hypot(dx, dy))
        crosstrack_steering_error, crosstrack_error = self.calculate_crosstrack_term(
            target_velocity, yaw, dx, dy, absolute_error
        )

        # Pure-Pursuit heading term at preview_index. Two stability
        # guards:
        #   (a) L_d_actual must be ≥ ~2 · wheelbase or the geometric
        #       formula δ = atan2(2L sinα, L_d) saturates on tiny α —
        #       it interprets a slightly-off preview point as needing
        #       a turn radius below the wheelbase, which is unphysical.
        #       The caller's `max(3.0, 0.5·v)` gate enforces this in
        #       arc-length terms; the explicit floor here protects
        #       against curved paths where the chord shrinks below
        #       the arc-length lookahead. First PP run saturated at
        #       startup (xte=−0.42, stanley=−25°) because L_d_actual
        #       was 0.48 m on a path that bent away from the front
        #       axle within the first 2 m.
        #   (b) Below ~0.5 m/s the chassis yaw rate ω = v·tan(δ)/L is
        #       negligible regardless of δ, so any heading command is
        #       indistinguishable from noise — but the slew-rate limit
        #       in control.py then carries that noisy command forward
        #       once the low-speed gate releases. Zero the heading
        #       term while nearly stationary; cross-track still applies.
        L_D_MIN_M = 2.0 * self.wheelbase  # wheelbase = 1.60 m → 3.2 m
        px_to_preview = self.px[preview_index] - fx
        py_to_preview = self.py[preview_index] - fy
        L_d_actual = float(np.hypot(px_to_preview, py_to_preview))
        if L_d_actual >= L_D_MIN_M and abs(target_velocity) >= 0.5:
            alpha = wrap_to_pi(atan2(py_to_preview, px_to_preview) - yaw)
            pp_heading_steer = atan2(
                2.0 * self.wheelbase * sin(alpha), L_d_actual
            )
        else:
            pp_heading_steer = 0.0

        yaw_rate_damping = self.calculate_yaw_rate_term(
            target_velocity, steering_angle
        )

        desired_steering_angle = (
            pp_heading_steer + crosstrack_steering_error + yaw_rate_damping
        )
        desired_steering_angle -= self.calculate_steering_delay_term(
            desired_steering_angle, steering_angle
        )
        limited_steering_angle = np.clip(
            desired_steering_angle, -self.max_steer, self.max_steer
        )

        return limited_steering_angle, target_index, crosstrack_error
