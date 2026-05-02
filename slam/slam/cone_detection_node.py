"""
========================
slam.py (v1.0)
========================

Elaborado por Jaime Perez para el ISC
Este nodo va separado del resto de slam para no cargar Numba para cada nodo

Contiene tres nodos:
1. Cone_Detection: Publica los resultados de final_cone_result_rt() este Nodo se puede mantener incendido y asi no hay que esperar
    a que compile cada vez que hay que probar. Numba tarda en optimizar el codigo y es tedioso hacerlo cada vez que se quiere probar.
"""

import sys
import os
import time
import numpy as np
import cv2 as cv
import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from slam.cone_detection import final_cone_result_rt, warmup_numba_functions

from fs_msgs.msg import Track, Cone

import cProfile, pstats, io

from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

QOS_LATEST = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class Cone_Detection(Node):
    """Publica los resultados de final_cone_result_rt() este Nodo se puede mantener encendido y asi no hay que esperar
    a que compile cada vez que hay que probar
    """

    # Cluster-height threshold separating big-orange cones (505 mm tall per
    # DS Table 1) from small blue/yellow/orange cones (325 mm). Live-measured
    # distributions at close range (start-gate sampling, 2026-04-19):
    #   small cones:  0.19–0.25 m
    #   big orange:   0.43–0.46 m
    # Gap of ~0.18 m between the two. Threshold sits in that gap, biased
    # toward the small-cone side so that long-range big cones — whose
    # measured height compresses with distance because fewer vertical LiDAR
    # channels hit them — still classify correctly.
    BIG_ORANGE_HEIGHT_THRESHOLD_M = 0.30

    def __init__(self):
        super().__init__("Cone_Detection")
        # Publicar
        self.publisher_MarkerArray = self.create_publisher(MarkerArray, "Conos_raw", 10)
        # Dedicated stream of big-orange cones (the FS finish-line markers).
        # Kept separate from /Conos_raw so SLAM's blue/yellow classifier
        # doesn't need to learn about orange, and so downstream consumers
        # (autonomous-stop logic in the control node) can subscribe without
        # filtering the whole cone list. Positions are in the car frame.
        self.publisher_Orange = self.create_publisher(MarkerArray, "Conos_Orange", 10)
        # Subscribir
        self.subscription = self.create_subscription(
            PointCloud2, "/fsds/lidar/Lidar1", self.listener_callback, QOS_LATEST
        )
        warmup_numba_functions()

        self.n_conos = 0
        # Per-second diagnostic accumulator for the cluster-filter pipeline.
        # Each LiDAR scan populates per-stage cluster counts via
        # final_cone_result_rt(debug_counters=...); we sum across scans and
        # log a single summary line per second. Helps localise where cones
        # are getting filtered out (issue #177).
        self._diag = {}
        self._diag_n_scans = 0
        self._diag_last_log_ns = 0

    def listener_callback(self, msg):
        # Parse PointCloud2 using point_step to handle any field layout (xyz, xyz+padding, xyzi, etc.)
        floats_per_point = msg.point_step // 4  # bytes per point / 4 bytes per float32
        num_points = msg.width * msg.height
        raw = np.frombuffer(msg.data, dtype=np.float32).reshape(num_points, floats_per_point)
        point_cloud = raw[:, :3]  # take only x, y, z

        conos = []
        per_scan_diag = {}
        try:  ###A veces da error de division por cero. El try: es para evitar que crashe
            conos = final_cone_result_rt(point_cloud, debug_counters=per_scan_diag)
        except Exception as e:
            import traceback
            self.get_logger().error(traceback.format_exc())
            pass

        # Pre-compute per-side tallies for the diagnostic. Cone_Detection
        # doesn't itself colour-classify (SLAM does that downstream), but
        # body-y sign is the same signal SLAM's classify() uses, so this
        # is a faithful proxy for "what colour SLAM will see for each
        # accepted cone." Used to determine whether a cone-imbalance in
        # the SLAM map (#189) starts here in detection or downstream.
        n_left = n_right = n_centerline = n_bigorange = 0
        for entry in conos:
            b = float(entry[1])  # body-y; +Y = left in REP-103
            h = float(entry[2]) if len(entry) >= 3 else 0.0
            if h > self.BIG_ORANGE_HEIGHT_THRESHOLD_M:
                n_bigorange += 1
            elif b > 0.5:
                n_left += 1
            elif b < -0.5:
                n_right += 1
            else:
                n_centerline += 1
        per_scan_diag["accepted_left"] = n_left
        per_scan_diag["accepted_right"] = n_right
        per_scan_diag["accepted_centerline"] = n_centerline
        per_scan_diag["accepted_bigorange"] = n_bigorange

        # Accumulate per-stage filter counts and log a per-second summary.
        for k, v in per_scan_diag.items():
            self._diag[k] = self._diag.get(k, 0) + v
        self._diag_n_scans += 1
        now_ns = self.get_clock().now().nanoseconds
        if self._diag_last_log_ns == 0:
            self._diag_last_log_ns = now_ns
        elif now_ns - self._diag_last_log_ns >= 1_000_000_000 and self._diag_n_scans > 0:
            n = self._diag_n_scans
            self.get_logger().info(
                f"CONE_FILTER (avg/scan over {n}): "
                f"pts={self._diag.get('n_input_points',0)/n:5.0f} "
                f"clusters={self._diag.get('n_clusters',0)/n:4.1f} "
                f"-> >3pts={self._diag.get('after_min_pts',0)/n:4.1f} "
                f"-> height={self._diag.get('after_height_gate',0)/n:4.1f} "
                f"-> fit/centroid={self._diag.get('after_fit_or_centroid',0)/n:4.1f} "
                f"(fit={self._diag.get('fit_used',0)/n:.1f}, "
                f"centroid={self._diag.get('centroid_used',0)/n:.1f}, "
                f"far_dropped={self._diag.get('far_dropped',0)/n:.1f}) "
                f"by-side: L={self._diag.get('accepted_left',0)/n:4.1f} "
                f"R={self._diag.get('accepted_right',0)/n:4.1f} "
                f"C={self._diag.get('accepted_centerline',0)/n:.1f} "
                f"BO={self._diag.get('accepted_bigorange',0)/n:.1f}"
            )
            self._diag = {}
            self._diag_n_scans = 0
            self._diag_last_log_ns = now_ns
        self.get_logger().debug(str(len(conos)))
        markerArray = MarkerArray()

        ###Aprovechar el metodo MarkerArray() para mandar resultados de final_cone_result_rt()
        orangeArray = MarkerArray()
        i = 0
        orange_i = 0
        for entry in conos:
            # Backward compat: legacy shapes were (x, y) and (x, y, height);
            # current shape is (x, y, height, sigma_xy).
            a = float(entry[0])
            b = float(entry[1])
            height = float(entry[2]) if len(entry) >= 3 else 0.0
            # sigma_xy < 0 is the sentinel for "uncertainty unknown" — SLAM
            # falls back to a range-only formula in that case.
            sigma_xy = float(entry[3]) if len(entry) >= 4 else -1.0
            is_big_orange = height > self.BIG_ORANGE_HEIGHT_THRESHOLD_M
            self.get_logger().debug(
                f"x: {a} Y: {b} h: {height:.2f} big_orange={is_big_orange}"
            )
            marker = Marker()
            marker.pose.position.x = a
            marker.pose.position.y = b
            marker.pose.position.z = 0.0

            ###Hacer compatible con RVIZ####
            marker.header.frame_id = "base_link"  ##El mapa esta en el sistema de referencia Odom no el coche
            marker.type = marker.CUBE
            if (
                i == 0
            ):  ##En el pimer elemeto se le dice a RVIZ que elimine los registros. Mas info en Wiki RVIZ MarkerArray
                marker.action = 3  # ELIMINAR TODO 3
            else:
                marker.action = marker.ADD  # Añadir marcardo

            marker.header.stamp = msg.header.stamp
            # marker.scale carries per-cone measurement metadata for
            # downstream SLAM, layered on top of the visualization-size
            # convention RViz/Foxglove expects:
            #   scale.x → σ_xy in metres (observation position uncertainty);
            #             negative sentinel means "unknown, use SLAM's
            #             range-only fallback"
            #   scale.y → reserved (was visualization width; kept default)
            #   scale.z → measured cluster height (existing convention,
            #             used to separate big-orange from small cones)
            # SLAM consumes scale.x via cone_graph_slam_node._observations
            # _from_markers; RViz happily renders cubes of σ-meter width.
            marker.scale.x = sigma_xy if sigma_xy > 0.0 else 0.1
            marker.scale.y = 0.1
            marker.scale.z = max(0.1, height)
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.pose.orientation.w = 1.0
            marker.id = i
            i += 1
            ###Hacer compatible con RVIZ####

            markerArray.markers.append(marker)

            if is_big_orange:
                # Same pose, distinct marker list, orange colour for RViz.
                orange = Marker()
                orange.header.frame_id = "base_link"
                orange.header.stamp = msg.header.stamp
                orange.type = Marker.CUBE
                orange.action = 3 if orange_i == 0 else Marker.ADD
                orange.pose.position.x = a
                orange.pose.position.y = b
                orange.pose.position.z = 0.0
                orange.pose.orientation.w = 1.0
                orange.scale.x = 0.3
                orange.scale.y = 0.3
                orange.scale.z = max(0.1, height)
                orange.color.a = 1.0
                orange.color.r = 1.0
                orange.color.g = 0.5
                orange.color.b = 0.0
                orange.id = orange_i
                orange_i += 1
                orangeArray.markers.append(orange)

        self.publisher_MarkerArray.publish(markerArray)
        # Publish even when the current scan has no big-orange cones so
        # downstream consumers don't keep a stale cache when the gate exits
        # the LiDAR FoV at close range (the sensor is mounted 1.4 m forward
        # of the car, so cones at the start gate leave the ±60° H-FOV before
        # the car has physically passed them).
        self.publisher_Orange.publish(orangeArray)


"""
Llamadas a Objetos para ROS2
"""


def cone_detection(args=None):
    rclpy.init(args=args)

    cone = Cone_Detection()
    rclpy.spin(cone)
