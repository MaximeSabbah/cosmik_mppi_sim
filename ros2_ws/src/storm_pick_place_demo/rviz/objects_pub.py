#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion

class ObjectsMarkers(Node):
    def __init__(self):
        super().__init__('objects_markers_pub')
        self.pub = self.create_publisher(MarkerArray, '/storm_pick_place/objects_markers', 10)
        self.timer = self.create_timer(1.0, self.tick)

        # Your objects
        self.objects = [
            dict(name='floor',  frame='fer_link0', pose=[0.4, 0.0, -0.05, 0, 0, 0, 1.0], dims=[1.4, 0.8, 0.01]),
            dict(name='brick1', frame='fer_link0', pose=[0.32, 0.0,  0.07, 0, 0, 0, 1.0], dims=[0.11, 0.22, 0.20]),
            dict(name='brick2', frame='fer_link0', pose=[0.63, 0.0,  0.09, 0, 0, 0, 1.0], dims=[0.11, 0.22, 0.20]),
            dict(name='box1',   frame='fer_link0', pose=[0.79, 0.0,  0.09, 0, 0, 0, 1.0], dims=[0.20, 0.35, 0.20]),
            dict(name='box2',   frame='fer_link0', pose=[0.47, 0.0,  0.08, 0, 0, 0, 1.0], dims=[0.20, 0.35, 0.20]),
        ]

    def make_marker(self, i, obj):
        m = Marker()
        m.header.frame_id = obj['frame']
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = 'objects'
        m.id = i
        m.type = Marker.CUBE
        m.action = Marker.ADD

        x, y, z, qx, qy, qz, qw = obj['pose']
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))

        sx, sy, sz = obj['dims']
        m.scale.x = float(sx)
        m.scale.y = float(sy)
        m.scale.z = float(sz)

        # Colors (change as you want)
        if obj['name'] == 'floor':
            m.color.r, m.color.g, m.color.b, m.color.a = 0.96, 0.71, 0.45, 1.0  # light maple, more yellow
        elif 'brick' in obj['name']:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.6, 0.1, 0.1, 1.0
        elif obj['name'] == 'box1':
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.7, 0.0, 1.0  # green
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.2, 0.6, 1.0

        # Keep markers alive; RViz will update when republished
        m.lifetime.sec = 0
        return m

    def tick(self):
        arr = MarkerArray()
        for i, obj in enumerate(self.objects):
            arr.markers.append(self.make_marker(i, obj))
        self.pub.publish(arr)

def main():
    rclpy.init()
    node = ObjectsMarkers()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

# from visualization_msgs.msg import MarkerArray

# class ObjectsMarkers(Node):
#     def __init__(self):
#         super().__init__('objects_markers_pub')

#         qos = QoSProfile(
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1,
#             reliability=ReliabilityPolicy.RELIABLE,
#             durability=DurabilityPolicy.TRANSIENT_LOCAL,
#         )

#         self.pub = self.create_publisher(
#             MarkerArray,
#             '/storm_pick_place/objects_markers',
#             qos
#         )

#         self.timer = self.create_timer(1.0, self.tick)

#     def tick(self):
#         msg = MarkerArray()
#         # fill msg.markers...
#         self.pub.publish(msg)

# def main():
#     rclpy.init()
#     node = ObjectsMarkers()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()