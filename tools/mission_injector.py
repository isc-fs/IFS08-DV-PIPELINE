#!/usr/bin/env python3
"""Mission Injector - Simulates uDV interface for pipeline testing.

Publishes /assi/state (AS state machine heartbeat) and /ami/mission
(mission selection) to allow the DV pipeline to run with rosbag replay
from manual driving data (which lacks mission control signals).
"""

import sys
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt8, Int32

# Mission name to AMI index mapping
# AMI index is what goes on /ami/mission (not mission_id!)
MISSION_NAME_TO_AMI_INDEX = {
    "trackdrive": 4,    # mission_id=1, ami_index=4
    "autocross": 3,     # mission_id=2, ami_index=3
    "acceleration": 1,  # mission_id=3, ami_index=1
    "skidpad": 2,       # mission_id=4, ami_index=2
}

# AS States (from interface_contract.py)
AS_OFF = 0
AS_EMERGENCY = 1
AS_READY = 2
AS_DRIVING = 3
AS_FINISHED = 4

# DV Status (from interface_contract.py)
DV_IDLE = 0
DV_PREPARING = 1
DV_READY = 2
DV_RUNNING = 3
DV_FINISHED = 4
DV_EMERGENCY = 5
DV_FAILED = 6

DV_STATUS_NAMES = {
    0: "IDLE", 1: "PREPARING", 2: "READY",
    3: "RUNNING", 4: "FINISHED", 5: "EMERGENCY", 6: "FAILED"
}


class MissionInjector(Node):
    def __init__(self, mission_name, auto_ready=False, auto_drive=False):
        super().__init__('mission_injector')
        
        self.mission_name = mission_name
        self.ami_index = MISSION_NAME_TO_AMI_INDEX.get(mission_name)
        if self.ami_index is None:
            valid = ", ".join(MISSION_NAME_TO_AMI_INDEX.keys())
            raise ValueError(f"Invalid mission '{mission_name}'. Valid: {valid}")
        
        self.auto_ready = auto_ready
        self.auto_drive = auto_drive
        self.current_as_state = AS_OFF
        self.dv_status = DV_IDLE
        self.start_time = time.time()
        
        # Publishers (10 Hz to satisfy watchdog)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_assi_state = self.create_publisher(UInt8, '/assi/state', qos)
        self.pub_ami_mission = self.create_publisher(Int32, '/ami/mission', qos)
        
        # Subscriber to monitor DV status
        self.sub_dv_status = self.create_subscription(
            UInt8, '/dv/status', self._on_dv_status, qos)
        
        # Timer for publishing (10 Hz = 100ms)
        self.timer = self.create_timer(0.1, self._publish_state)
        
        # Input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
        self.get_logger().info(f"Mission Injector started")
        self.get_logger().info(f"Mission: {mission_name} (AMI index: {self.ami_index})")
        self.get_logger().info(f"Auto-ready: {auto_ready}, Auto-drive: {auto_drive}")
        self.get_logger().info("")
        self.get_logger().info("Commands: r=ready, d=drive, f=finish, e=emergency, o=off, q=quit")
    
    def _on_dv_status(self, msg):
        self.dv_status = msg.data
        status_name = DV_STATUS_NAMES.get(self.dv_status, "UNKNOWN")
        
        # Auto-drive: transition to DRIVING when DV becomes READY
        if self.auto_drive and self.dv_status == DV_READY and self.current_as_state == AS_READY:
            self.get_logger().info(f"[AUTO] DV is READY, transitioning to DRIVING")
            self.current_as_state = AS_DRIVING
    
    def _publish_state(self):
        # Auto-ready: transition to READY after 5 seconds
        if self.auto_ready and self.current_as_state == AS_OFF:
            if time.time() - self.start_time > 5.0:
                self.get_logger().info("[AUTO] Transitioning to READY")
                self.current_as_state = AS_READY
        
        # Publish AS state and mission
        msg_state = UInt8()
        msg_state.data = self.current_as_state
        self.pub_assi_state.publish(msg_state)
        
        msg_mission = Int32()
        msg_mission.data = self.ami_index
        self.pub_ami_mission.publish(msg_mission)
    
    def _input_loop(self):
        """Non-blocking input loop in separate thread."""
        while rclpy.ok():
            try:
                cmd = input().strip().lower()
                if cmd == 'r':
                    self.get_logger().info("→ AS_READY")
                    self.current_as_state = AS_READY
                elif cmd == 'd':
                    self.get_logger().info("→ AS_DRIVING")
                    self.current_as_state = AS_DRIVING
                elif cmd == 'f':
                    self.get_logger().info("→ AS_FINISHED")
                    self.current_as_state = AS_FINISHED
                elif cmd == 'e':
                    self.get_logger().info("→ AS_EMERGENCY")
                    self.current_as_state = AS_EMERGENCY
                elif cmd == 'o':
                    self.get_logger().info("→ AS_OFF")
                    self.current_as_state = AS_OFF
                elif cmd == 'q':
                    self.get_logger().info("Shutting down...")
                    rclpy.shutdown()
                    break
                elif cmd:
                    self.get_logger().warn(f"Unknown command: {cmd}")
            except EOFError:
                break
            except Exception as e:
                self.get_logger().error(f"Input error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mission Injector for pipeline testing")
    parser.add_argument('--mission', required=True, 
                       choices=list(MISSION_NAME_TO_AMI_INDEX.keys()),
                       help="Mission name")
    parser.add_argument('--auto-ready', action='store_true',
                       help="Automatically transition to READY after 5s")
    parser.add_argument('--auto-drive', action='store_true',
                       help="Automatically transition to DRIVING when DV is READY")
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        node = MissionInjector(args.mission, args.auto_ready, args.auto_drive)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
