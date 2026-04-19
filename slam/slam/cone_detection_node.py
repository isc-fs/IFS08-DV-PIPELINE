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
    # DS Table 1) from small blue/yellow/orange cones (325 mm). The midpoint
    # leaves generous margin for LiDAR vertical-quantization noise.
    BIG_ORANGE_HEIGHT_THRESHOLD_M = 0.4

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

    def listener_callback(self, msg):
        # Parse PointCloud2 using point_step to handle any field layout (xyz, xyz+padding, xyzi, etc.)
        floats_per_point = msg.point_step // 4  # bytes per point / 4 bytes per float32
        num_points = msg.width * msg.height
        raw = np.frombuffer(msg.data, dtype=np.float32).reshape(num_points, floats_per_point)
        point_cloud = raw[:, :3]  # take only x, y, z

        conos = []
        try:  ###A veces da error de division por cero. El try: es para evitar que crashe
            conos = final_cone_result_rt(point_cloud)
        except Exception as e:
            import traceback
            self.get_logger().error(traceback.format_exc())
            pass
        self.get_logger().debug(str(len(conos)))
        markerArray = MarkerArray()

        ###Aprovechar el metodo MarkerArray() para mandar resultados de final_cone_result_rt()
        orangeArray = MarkerArray()
        i = 0
        orange_i = 0
        for entry in conos:
            # Backward compat: old return shape was (x, y); new is (x, y, height).
            if len(entry) >= 3:
                a, b, height = float(entry[0]), float(entry[1]), float(entry[2])
            else:
                a, b = float(entry[0]), float(entry[1])
                height = 0.0
            is_big_orange = height > self.BIG_ORANGE_HEIGHT_THRESHOLD_M
            self.get_logger().debug(
                f"x: {a} Y: {b} h: {height:.2f} big_orange={is_big_orange}"
            )
            marker = Marker()
            marker.pose.position.x = a
            marker.pose.position.y = b
            marker.pose.position.z = 0.0

            ###Hacer compatible con RVIZ####
            marker.header.frame_id = "fsds/FSCar"  ##El mapa esta en el sistema de referencia Odom no el coche
            marker.type = marker.CUBE
            if (
                i == 0
            ):  ##En el pimer elemeto se le dice a RVIZ que elimine los registros. Mas info en Wiki RVIZ MarkerArray
                marker.action = 3  # ELIMINAR TODO 3
            else:
                marker.action = marker.ADD  # Añadir marcardo

            marker.header.stamp = msg.header.stamp
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            # Encode measured cluster height on scale.z so downstream consumers
            # can distinguish big orange from small without re-measuring.
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
                orange.header.frame_id = "fsds/FSCar"
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
