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
        # global_cones, car_position, car_direction = load_data()

    def listener_callback(self, msg):
        self.mapa = msg
        # global_cones, car_position, car_direction = load_data()
        # self.path_planner = PathPlanner(MissionTypes.trackdrive)
        global_cones = [numpy.zeros((0, 2)) for _ in range(5)]
        l = []
        for marker in self.mapa.markers:

            # self.get_logger().info('jajaj'+str(marker.pose.position.x)+'ajaj'+str(marker.pose.position.y)+str(type(marker.pose.position.x)))
            l.append(
                (
                    float(marker.pose.position.x) + 0.0,
                    float(marker.pose.position.y) + 0.0,
                )
            )
            pass

        # print(numpy.array(l))

        global_cones[ConeTypes.UNKNOWN] = numpy.array(l)
        # print(global_cones)

        # global_cones is a sequence that contains 5 numpy arrays with shape (N, 2),
        # where N is the number of cones of that type

        # ConeTypes is an enum that contains the following values:
        # ConeTypes.UNKNOWN which maps to index 0
        # ConeTypes.RIGHT/ConeTypes.YELLOW which maps to index 1
        # ConeTypes.LEFT/ConeTypes.BLUE which maps to index 2
        # ConeTypes.START_FINISH_AREA/ConeTypes.ORANGE_SMALL which maps to index 3
        # ConeTypes.START_FINISH_LINE/ConeTypes.ORANGE_BIG which maps to index 4

        try:  ###Generar Objeto de transformada entre Odom y el coche
            t = self.tf_buffer.lookup_transform(
                "odom", "fsds/FSCar", rclpy.time.Time()
            )  ###Revisar tiempo
        except TransformException as ex:
            print(ex)  ##Error al optener TF
            return

        try:  ###Generar Objeto de transformada entre Odom y el coche
            t_inv = self.tf_buffer.lookup_transform(
                "fsds/FSCar", "odom", rclpy.time.Time()
            )  ###Revisar tiempo
        except TransformException as ex:
            print(ex)  ##Error al optener TF
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
        # car_position=numpy.array([0.0,0.0])
        # self.get_logger().info('jajaj'+str(car_position[0])+'ajaj'+str(car_position[1]))
        car_direction = numpy.array([numpy.cos(yaw), numpy.sin(yaw)])

        # car_position is a 2D numpy array with shape (2,)
        # car_direction is a 2D numpy array with shape (2,) representing the car's direction vector
        # car_direction can also be a float representing the car's direction in radians

        # print("inincio")
        path = self.path_planner.calculate_path_in_global_frame(
            global_cones, car_position, car_direction
        )
        # print("fin")
        # path is a Mx4 numpy array, where M is the number of points in the path
        # the columns represent the spline parameter (distance along path), x, y and path curvature
        s = []
        x = []
        y = []
        for a, b, c, d in path:
            s.append(a)
            x.append(b)
            y.append(c)

        s_dense = numpy.linspace(s[0], s[-1], num=30)
        px = numpy.interp(s_dense, s, x)
        py = numpy.interp(s_dense, s, y)
        dx_ds = numpy.gradient(px, s_dense)
        dy_ds = numpy.gradient(py, s_dense)
        pyaw = numpy.arctan2(dy_ds, dx_ds)

        track = Path()
        track.header.frame_id = "odom"
        for i, x in enumerate(px):
            track.poses.append(gen_mark(float(x), float(py[i]), pyaw[i]))

        self.publisher_path.publish(track)


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
