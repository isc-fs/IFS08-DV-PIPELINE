import sys
import os
import time
import numpy

import fs_msgs as fsds

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs


class Publicar_TF(Node):
    def __init__(self):
        super().__init__("Publicar_Odom")
        # Publicacion
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscricion
        self.posicion = self.create_subscription(
            Odometry,
            "/fsds/testing_only/odom",  ##Mensaje de Odometria---Mas adelante cambiar a estimacion VREL
            self.listener_callback,
            10,
        )

    def listener_callback(self, odom):
        msg = Odometry()
        t = TransformStamped()

        #######Publicar Transformadad#######

        t.header.stamp = odom.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "fsds/FSCar"

        t.transform.translation.x = odom.pose.pose.position.x
        t.transform.translation.y = odom.pose.pose.position.y
        t.transform.translation.z = 0.0

        t.transform.rotation.x = odom.pose.pose.orientation.x
        t.transform.rotation.y = odom.pose.pose.orientation.y
        t.transform.rotation.z = odom.pose.pose.orientation.z
        t.transform.rotation.w = odom.pose.pose.orientation.w

        self.tf_broadcaster.sendTransform(t)


def TF(args=None):
    rclpy.init(args=args)

    Publicar_odom = Publicar_TF()
    rclpy.spin(Publicar_odom)
