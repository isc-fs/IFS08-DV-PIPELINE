"""
========================
path_planning.py (v1.0)
========================

Elaborado por Jaime Perez para el ISC
Permite conectar el simulador FSDS a ROS2 mediante la API de Python.
Publica Odometria(/odom), TF coche-odom,
Datos de Lidar en formato nube de puntos(3D)(/cloud_in) y LaserScan(2D)(/scan)
La conversion de nube de puntos a Laser se hace con el paquete pointcloud-to-laserscan que se deve instalar(readme.md)
"""

import sys
import os
import time
import numpy
import cv2 as cv

from math import atan2, pi, sqrt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TwistWithCovarianceStamped

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import Point
import geometry_msgs

from fs_msgs.msg import ControlCommand
from fs_msgs.srv import Reset

from transforms3d.euler import quat2euler, euler2quat

from scipy import interpolate
from sklearn.neighbors import KDTree

# from path_planning.fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes


def gen_mark(x, y, yaw):
    mark = PoseStamped()
    mark.header.frame_id = "odom"

    mark.pose.position.x = x
    mark.pose.position.y = y
    mark.pose.position.z = 0.0

    a = euler2quat(0, 0, yaw)
    mark.pose.orientation.w = a[0]
    mark.pose.orientation.x = a[1]
    mark.pose.orientation.y = a[2]
    mark.pose.orientation.z = a[3]

    # (mark.pose.orientation.x,mark.pose.orientation.y,mark.pose.orientation.z,mark.pose.orientation.w)=(1.0,0.0,0.0,-1.0)#euler2quat(0,0,0)

    return mark


class Plan_Path(Node):
    def __init__(self):
        super().__init__("Plan_Path")
        # Publicar
        self.publisher_path = self.create_publisher(Path, "Path", 10)
        # Subscricion
        self.subscription_conos = self.create_subscription(
            MarkerArray, "Conos", self.listener_callback, 10
        )
        self.mapa = MarkerArray()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_planner = PathPlanner(MissionTypes.trackdrive)

        # Pre-warm numba JIT with dummy cones so the first live callback isn't
        # blocked for 30+ s of compilation. While JIT is running, Python can't
        # service SIGTERM; if the pipeline is stopped mid-JIT (Start Session
        # stops and restarts the pipeline), the launcher escalates to SIGKILL
        # and Plan_Path dies with exit code -9. Paying the JIT cost here keeps
        # the spin loop responsive from the first message onward.
        try:
            warmup_cones = [numpy.zeros((0, 2)) for _ in range(5)]
            warmup_cones[ConeTypes.LEFT] = numpy.array([[0.0, 1.5], [5.0, 1.5]])
            warmup_cones[ConeTypes.RIGHT] = numpy.array([[0.0, -1.5], [5.0, -1.5]])
            self.path_planner.calculate_path_in_global_frame(
                warmup_cones, numpy.array([0.0, 0.0]), numpy.array([1.0, 0.0])
            )
            self.get_logger().info("Plan_Path: numba JIT warmup complete")
        except Exception as ex:
            # Warmup failure is non-fatal — we'd rather launch degraded than not at all.
            self.get_logger().warn(f"Plan_Path: JIT warmup failed (non-fatal): {ex}")
        # global_cones, car_position, car_direction = load_data()

        # Instrumentation for the curve-failure audit. Each callback falls
        # into exactly one of these buckets; per-second rates are logged
        # in _maybe_log_stats. A healthy run should be cb≈pub (every
        # callback publishes) — gate_miss > 0 directly proves the
        # forward-cone gate is starving the controller during the corner.
        self._stats = {
            "callbacks": 0,
            "low_cones":   0,        # < 2 total cones in /Conos this tick
            "gate_miss_L": 0,        # left_ahead < threshold
            "gate_miss_R": 0,        # right_ahead < threshold
            "gate_miss_both": 0,     # both sides starved
            "tf_miss":     0,        # TF lookup failed
            "plan_excpt":  0,        # fsd_path_planning raised
            "plan_short":  0,        # planner returned < 2 points
            "publish":     0,        # fresh path emitted
        }
        self._stats_prev = dict(self._stats)
        self._stats_last_log_ns = 0
        self._window_min_L_ahead = 999
        self._window_min_R_ahead = 999

    def _maybe_log_stats(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._stats_last_log_ns == 0:
            self._stats_last_log_ns = now_ns
            return
        if now_ns - self._stats_last_log_ns < 1_000_000_000:
            return
        dt = (now_ns - self._stats_last_log_ns) / 1e9
        d = {k: self._stats[k] - self._stats_prev[k] for k in self._stats}
        miss_total = d["gate_miss_L"] + d["gate_miss_R"] + d["gate_miss_both"]
        # min_L/min_R show the worst-case margin in the window. Sentinel
        # 999 means "no callback in the window had cone counts to record"
        # — render as 'n/a' to avoid confusion with a real high count.
        min_L = "n/a" if self._window_min_L_ahead == 999 else str(self._window_min_L_ahead)
        min_R = "n/a" if self._window_min_R_ahead == 999 else str(self._window_min_R_ahead)
        self.get_logger().info(
            f"PATH_RATE cb={d['callbacks']/dt:4.1f}/s pub={d['publish']/dt:4.1f}/s "
            f"gate_miss={miss_total/dt:4.1f}/s (L={d['gate_miss_L']} R={d['gate_miss_R']} "
            f"both={d['gate_miss_both']}) min_L={min_L} min_R={min_R} "
            f"low={d['low_cones']} tf={d['tf_miss']} "
            f"excpt={d['plan_excpt']} short={d['plan_short']}"
        )
        self._stats_prev = dict(self._stats)
        self._stats_last_log_ns = now_ns
        self._window_min_L_ahead = 999
        self._window_min_R_ahead = 999

    def listener_callback(self, msg):
        self._stats["callbacks"] += 1
        self.mapa = msg
        global_cones = [numpy.zeros((0, 2)) for _ in range(5)]
        left_cones = []   # blue (ConeTypes.LEFT)
        right_cones = []  # yellow (ConeTypes.RIGHT)
        unknown_cones = []

        for marker in self.mapa.markers:
            if marker.action == 3:  # DELETEALL marker — skip, no real cone position
                continue
            x = float(marker.pose.position.x)
            y = float(marker.pose.position.y)
            r = marker.color.r
            g = marker.color.g
            b = marker.color.b
            if b > 0.8 and r < 0.2 and g < 0.2:
                left_cones.append((x, y))
            elif r > 0.8 and g > 0.8 and b < 0.2:
                right_cones.append((x, y))
            else:
                unknown_cones.append((x, y))

        if left_cones:
            global_cones[ConeTypes.LEFT] = numpy.array(left_cones)
        if right_cones:
            global_cones[ConeTypes.RIGHT] = numpy.array(right_cones)
        if unknown_cones:
            global_cones[ConeTypes.UNKNOWN] = numpy.array(unknown_cones)

        total_cones = len(left_cones) + len(right_cones) + len(unknown_cones)
        if total_cones < 2:
            self._stats["low_cones"] += 1
            self._maybe_log_stats()
            return

        # global_cones is a sequence that contains 5 numpy arrays with shape (N, 2),
        # where N is the number of cones of that type

        # ConeTypes is an enum that contains the following values:
        # ConeTypes.UNKNOWN which maps to index 0
        # ConeTypes.RIGHT/ConeTypes.YELLOW which maps to index 1
        # ConeTypes.LEFT/ConeTypes.BLUE which maps to index 2
        # ConeTypes.START_FINISH_AREA/ConeTypes.ORANGE_SMALL which maps to index 3
        # ConeTypes.START_FINISH_LINE/ConeTypes.ORANGE_BIG which maps to index 4

        try:
            t = self.tf_buffer.lookup_transform("odom", "fsds/FSCar", rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            self._stats["tf_miss"] += 1
            self._maybe_log_stats()
            return

        try:
            t_inv = self.tf_buffer.lookup_transform("fsds/FSCar", "odom", rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f"TF inverse lookup failed: {ex}")
            self._stats["tf_miss"] += 1
            self._maybe_log_stats()
            return

        yaw = quat2euler(
            [
                t.transform.rotation.w,
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
            ]
        )[2]

        car_position = numpy.array(
            [t.transform.translation.x, t.transform.translation.y]
        )
        car_direction = numpy.array([numpy.cos(yaw), numpy.sin(yaw)])

        # Forward-cone gate. fsd_path_planning produces a degenerate path
        # when the visible cones are all at or behind the car's longitudinal
        # position — its cone-walking heuristic falls back to the cross-
        # track LEFT–RIGHT pairing axis. At session start the car spawns
        # inside the start gate, the only cones in view are the four big-
        # orange corners straddling the car, and the resulting path comes
        # out 90° rotated (track goes north, planner emits east — DIAG
        # captured xte=+4.6 m, yaw_err=−90° for several seconds before
        # forward cones came into view). Require ≥ 2 LEFT + 2 RIGHT cones
        # ahead of the car (vehicle-frame X > 0) before publishing; while
        # the gate is closed the previous /Path stays in effect (or the
        # control loop sees forward_path_points empty and brakes neutral).
        left_arr = global_cones[ConeTypes.LEFT]
        right_arr = global_cones[ConeTypes.RIGHT]

        def _count_ahead(arr):
            if arr.shape[0] == 0:
                return 0
            fwd = (
                (arr[:, 0] - car_position[0]) * car_direction[0]
                + (arr[:, 1] - car_position[1]) * car_direction[1]
            )
            return int((fwd > 0.0).sum())

        # Dropped 2 → 1 on 2026-04-26 after the color-cache patch run
        # showed sustained gate-miss starvation (gate_miss=10/s, all R-side)
        # late in the corner — the path stopped publishing for ~3s while
        # the car drifted off. With color-classification stable from the
        # cache, a single forward cone per side is enough to define a
        # corridor edge; the planner can pair it with whatever else is
        # visible. The min_L/min_R counts in the PATH_RATE log fingerprint
        # whether this threshold drop covered the failing case (min_R=1)
        # or whether even 1 was too high (min_R=0 — needs upstream fix).
        MIN_AHEAD_PER_SIDE = 1
        left_ahead = _count_ahead(left_arr)
        right_ahead = _count_ahead(right_arr)
        self._window_min_L_ahead = min(self._window_min_L_ahead, left_ahead)
        self._window_min_R_ahead = min(self._window_min_R_ahead, right_ahead)
        if left_ahead < MIN_AHEAD_PER_SIDE or right_ahead < MIN_AHEAD_PER_SIDE:
            l_short = left_ahead < MIN_AHEAD_PER_SIDE
            r_short = right_ahead < MIN_AHEAD_PER_SIDE
            if l_short and r_short:
                self._stats["gate_miss_both"] += 1
            elif l_short:
                self._stats["gate_miss_L"] += 1
            else:
                self._stats["gate_miss_R"] += 1
            # Per-tick gate-miss detail. UNKNOWN bucket includes 'ref'-tagged
            # cones (RGB white from generar_trazas) and any cone the spatial
            # classifier couldn't tag — diagnostic for whether allowing the
            # planner to use UNKNOWN cones in the gate would rescue the run.
            unknown_ahead = _count_ahead(global_cones[ConeTypes.UNKNOWN])
            self.get_logger().info(
                f"GATE_MISS L={left_ahead} R={right_ahead} U={unknown_ahead} "
                f"total_ahead={left_ahead + right_ahead + unknown_ahead}"
            )
            self._maybe_log_stats()
            return

        try:
            path = self.path_planner.calculate_path_in_global_frame(
                global_cones, car_position, car_direction
            )
        except Exception as ex:
            self.get_logger().warn(f"Path calculation failed: {ex}")
            self._stats["plan_excpt"] += 1
            self._maybe_log_stats()
            return

        if path is None or len(path) < 2:
            self.get_logger().warn("Path planner returned empty path — skipping publish")
            self._stats["plan_short"] += 1
            self._maybe_log_stats()
            return

        s, x, y = [], [], []
        for a, b, c, _d in path:
            s.append(a)
            x.append(b)
            y.append(c)

        s_dense = numpy.linspace(s[0], s[-1], num=30)
        px = numpy.interp(s_dense, s, x)
        py = numpy.interp(s_dense, s, y)

        # Temporal path smoothing was tested at α=0.5 (same-index blend
        # with the previous published path) but the trajectory recorder
        # showed a consistent leftward drift on a straight section of
        # track even though the latest /Path was correct: the smoothed
        # first points lag ~0.5–1 m behind the car, biasing the PP
        # preview-point bearing one direction over many ticks. Reverted
        # to publishing the new path crisply each callback. Path-noise
        # mitigation has to live downstream (controller filtering) or
        # be done in arc-length space, not by blending world positions.
        dx_ds = numpy.gradient(px, s_dense)
        dy_ds = numpy.gradient(py, s_dense)
        pyaw = numpy.arctan2(dy_ds, dx_ds)

        track = Path()
        track.header.frame_id = "odom"
        for i, xi in enumerate(px):
            track.poses.append(gen_mark(float(xi), float(py[i]), pyaw[i]))

        self.publisher_path.publish(track)
        self._stats["publish"] += 1
        self._maybe_log_stats()


class Reser_server(Node):

    def __init__(self):
        super().__init__("Servicio_reset")
        self.cli = self.create_client(Reset, "reset")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("El sevicio no esta disponible")
        self.req = Reset.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


"""
Llamadas a Objetos para ROS2
"""


def plan_path(args=None):
    rclpy.init(args=args)

    Laser_stam = Plan_Path()
    rclpy.spin(Laser_stam)


def reiniciar(args=None):
    rclpy.init(args=args)

    minimal_client = Reser_server()
    minimal_client.send_request()
    minimal_client.destroy_node()
    rclpy.shutdown()
