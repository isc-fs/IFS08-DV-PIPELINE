"""
========================
slam.py (v1.0)
========================

Elaborado por Jaime Perez para el ISC
Contiene las funciones para realizar el Slam.
Se ha decidido separar este codfio de la parte del puente_ros para simplificar el codigo y mejorar el paralelizacion.

Contiene tres nodos:
1. Publicar_Mapa: Se suscribe al nodo anterior y añade esos conos a un mapa de features (mapa.py-mas detelles). Luego publica el mapa
    entero con un MarkerArray a RVIZ.

1. Publicar_Track: Publica la posicion real de los conos del track

3. Publicar_Laser(EXPERIMENTO): Pretende publicar como un escaneo de laser los reslutados de Cone_Detection. Para luego introducirlo en SlamToolBox
"""

import sys
import os
import time
import numpy
import cv2 as cv
import math

import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point, PointStamped
from tf2_geometry_msgs import do_transform_point

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose

from slam.mapa import *

from fs_msgs.msg import Track, Cone
from fs_msgs.srv import Reset

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class Publicar_Mapa(Node):
    """Publica el mapa"""

    def __init__(self):
        super().__init__("Publicar_Mapa")
        # Publicar
        self.publisher_MarkerArray = self.create_publisher(MarkerArray, "Conos", 10)
        self.publisher_Path_azul = self.create_publisher(Path, "Track_azul", 10)
        self.publisher_Path_amarillo = self.create_publisher(Path, "Track_amarillo", 10)

        # Subscripciones
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.subscription = self.create_subscription(
            MarkerArray, "Conos_raw", self.listener_callback, 10
        )

        # Servicio de reset
        self.srv = self.create_service(Reset, "reset", self.reset_callback)

        # Iniciar calse Mapa
        self.mapa = Mapa()

        # Color cache. `mapa.actualizar_mapa` rebuilds `self.conos` from
        # scratch every tick with `color='ns'`, and the spatial classifier
        # below would otherwise re-decide blue/yellow purely from the
        # current vehicle-frame Y of each cone — which flips for cones
        # near the longitudinal axis as the car rotates through a corner.
        # Direct cause of unstable /Path R values (audit 2026-04-26: R
        # bouncing 220m → 8m → 3m → 0.9m within 3s during a curve attempt).
        #
        # Cache strategy: keyed by world position (snap radius 0.5 m, well
        # within DBSCAN cluster drift ~10 cm and far below the FSG 3 m min
        # cone spacing). First classification wins; later ticks reuse the
        # cached color regardless of where the cone falls in vehicle frame.
        # Cleared on /reset. Each entry is (world_x, world_y, color).
        self._color_cache = []
        self._color_cache_radius_sq = 0.5 ** 2

        # Diagnostic counters for the cache, logged once per second.
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_last_log_ns = 0

    def _lookup_or_classify(self, world_x: float, world_y: float, p_rel_y: float) -> str:
        """Return the cached color for the cone at (world_x, world_y), or
        classify it from `p_rel_y` (sign-of-Y rule, vehicle frame REP-103),
        cache the result, and return it. Linear scan over the cache — for
        FSD tracks (≤300 cones) this is cheaper than a KDTree rebuild."""
        rsq = self._color_cache_radius_sq
        for cx, cy, color in self._color_cache:
            if (world_x - cx) ** 2 + (world_y - cy) ** 2 < rsq:
                self._cache_hits += 1
                return color
        self._cache_misses += 1
        color = 'Azul' if p_rel_y > 0.0 else 'Amarillo'
        self._color_cache.append((world_x, world_y, color))
        return color

    def _maybe_log_cache_stats(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._cache_last_log_ns == 0:
            self._cache_last_log_ns = now_ns
            return
        if now_ns - self._cache_last_log_ns < 1_000_000_000:
            return
        total = self._cache_hits + self._cache_misses
        if total > 0:
            self.get_logger().info(
                f"COLOR_CACHE size={len(self._color_cache)} "
                f"hits={self._cache_hits} misses={self._cache_misses} "
                f"hit_rate={100.0 * self._cache_hits / total:.0f}%"
            )
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_last_log_ns = now_ns

    def reset_callback(self, request, response):
        self.mapa.conos = []
        self.mapa.deteciones = []
        self._color_cache = []
        self.get_logger().info("Reseteando Mapa (color cache cleared)")
        return response

    def listener_callback(self, msg):
        if len(msg.markers) == 0:  ###Si no se han detectado conos parar
            return

        try:  ###Generar Objeto de transformada entre Odom y el coche
            t = self.tf_buffer.lookup_transform(
                "odom", "fsds/FSCar", rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return

        try:  ###Generar Objeto de transformada entre coche y odom. Transformada inversa
            t_inv = self.tf_buffer.lookup_transform(
                "fsds/FSCar", "odom", rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF inverse lookup failed: {ex}")
            return

        for (
            mark
        ) in (
            msg.markers
        ):  ###Añadir conos que detectado final_cone_result_rt() a el mapa
            self.mapa.add_detecion(mark.pose.position.x, mark.pose.position.y, t, t_inv)

        self.mapa.actualizar_mapa()
        self.mapa.generar_trazas(t, t_inv)

        # Color assignment, applied AFTER generar_trazas. First time we see
        # a cone (cache miss) we classify it by sign of vehicle-frame Y
        # (REP-103: +Y = left of car). On subsequent ticks the color cache
        # returns the already-decided color regardless of the cone's
        # current vehicle-frame Y — this is the fix for the corner-flip
        # failure where R (path radius) was oscillating 220m→8m→0.9m as
        # cones near the longitudinal axis flipped sides on every tick.
        # 'ref' cones (the two reference picks from generar_trazas) keep
        # their ref tag for visualisation and are not classified here.
        for cono in self.mapa.conos:
            if cono.color == 'ref':
                continue
            try:
                p = Point(x=cono.x, y=cono.y, z=0.0)
                p_rel = do_transform_point(PointStamped(point=p), t_inv).point
            except Exception:
                continue
            cono.color = self._lookup_or_classify(cono.x, cono.y, p_rel.y)
        self._maybe_log_cache_stats()

        markerArray = MarkerArray()

        ###Mostrar Conos con Marker Array###
        # Eliminar marcadores anterioires
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CUBE
        marker.action = 3  # ELIMINAR TODO 3
        marker.id = 0
        markerArray.markers.append(marker)

        for i, cono in enumerate(self.mapa.conos):  ###Mostrar el mapa completo
            marker = Marker()
            marker.header.frame_id = (
                "odom"  ##El mapa esta en el sistema de referencia Odom no el coche
            )
            marker.type = marker.MESH_RESOURCE
            marker.action = marker.ADD  # Añadir marcardo

            ##Tamaño de m
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0

            # Hay que incluir la referencia en setup.py para que colcon añada al ejecutable la carpeta de meches
            marker.mesh_resource = "package://slam/meshes/any_small.dae"

            if cono.color == "ref":
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0

            if cono.color == "cont":
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0

            ##Color
            if cono.color == "Azul":
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0
            elif cono.color == "Amarillo":  # Amarillo
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

            elif cono.color == "ns":  # No se sabe
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0

            ##Posicion
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = cono.x
            marker.pose.position.y = cono.y
            marker.pose.position.z = 0.0
            marker.id = i + 1

            markerArray.markers.append(marker)

        self.publisher_MarkerArray.publish(markerArray)

        ###Mostrar Track###
        ###Azul###
        track = Path()
        track.header.frame_id = "odom"
        for i, pose in enumerate(self.mapa.track_azul):
            mark = PoseStamped()
            mark.header.frame_id = "odom"

            mark.pose.position.x = pose.x
            mark.pose.position.y = pose.y
            mark.pose.position.z = 0.0

            mark.pose.orientation.x = 0.0
            mark.pose.orientation.y = 0.0
            mark.pose.orientation.z = 0.0
            mark.pose.orientation.w = 0.0

            track.poses.append(mark)

        self.publisher_Path_azul.publish(track)

        ###Amarillo###
        track = Path()
        track.header.frame_id = "odom"
        for i, pose in enumerate(self.mapa.track_amarillo):
            mark = PoseStamped()
            mark.header.frame_id = "odom"

            mark.pose.position.x = pose.x
            mark.pose.position.y = pose.y
            mark.pose.position.z = 0.0

            mark.pose.orientation.x = 0.0
            mark.pose.orientation.y = 0.0
            mark.pose.orientation.z = 0.0
            mark.pose.orientation.w = 0.0

            track.poses.append(mark)

        self.publisher_Path_amarillo.publish(track)


class Publicar_Track(Node):
    def __init__(self):
        super().__init__("Publicar_Laser")
        # Publicar
        self.publisher_MarkerArray = self.create_publisher(MarkerArray, "Track", 10)
        # Subscripcion
        self.subscription = self.create_subscription(
            Track, "/fsds/testing_only/track", self.listener_callback, 10
        )

    def listener_callback(self, msg):
        Cone_list = MarkerArray()
        i = 0
        for cone in msg.track:

            marker = Marker()
            marker.header.frame_id = (
                "odom"  ##El mapa esta en el sistema de referencia Odom no el coche
            )
            marker.type = marker.CUBE
            if (
                i == 0
            ):  ##En el pimer elemeto se le dice a RVIZ que elimine los registros. Mas info en Wiki RVIZ MarkerArray
                marker.action = 3  # ELIMINAR TODO 3
            else:
                marker.action = marker.ADD  # Añadir marcardo

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = cone.location.x
            marker.pose.position.y = cone.location.y
            marker.pose.position.z = 0.0
            marker.id = i
            i += 1

            Cone_list.markers.append(marker)

        self.publisher_MarkerArray.publish(Cone_list)
        print(len(Cone_list.markers))


class BenchMark(Node):
    def __init__(self):
        super().__init__("BenchMark_Slam")

        self.subscription = self.create_subscription(
            MarkerArray, "Conos", self.listener_callback, 10
        )
        self.len_conos = 0

        self.subscription = self.create_subscription(
            MarkerArray, "Track", self.listener_callback_track, 10
        )
        self.len_conos = 0

    def listener_callback_track(self, msg):
        # Cone_list = MarkerArray()
        self.len_conos = len(msg.markers)

        """for cone in msg.track:
            cone.location.x
            cone.location.y"""

    def listener_callback(self, msg):
        # self.get_logger().info('n_detectados')
        # self.get_logger().info(str(len(msg.markers)))
        # self.get_logger().info('n_real')
        # self.get_logger().info(str(self.len_conos))
        pass


"""
Llamadas a Objetos para ROS2
"""


def publicar_mapa(args=None):
    rclpy.init(args=args)

    mapa = Publicar_Mapa()
    rclpy.spin(mapa)


def BenchMark_Slam(args=None):
    rclpy.init(args=args)

    BenchMark_slam = BenchMark()
    rclpy.spin(BenchMark_slam)


def publicar_track(args=None):
    rclpy.init(args=args)

    nodo_laser = Publicar_Track()
    rclpy.spin(nodo_laser)
