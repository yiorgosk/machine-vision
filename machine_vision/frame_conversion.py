import rclpy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
import pandas as pd
import csv
import sys
import os


class Subscriber(Node):

    def __init__(self):
        super().__init__('tf_converter')
        qos_profile = QoSProfile(depth=10)

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

        self.timer = self.create_timer(0.5, self.timer_callback)
        self.get_logger().info('Starting {}'.format(self.get_name()))
        self.i = 0
        self.initial_point = PointStamped()
        self.transformed_point = 0
        self.data = pd.read_csv('file1.csv', delimiter=';')
        self.line_number = self.line_count('file1.csv')


    def timer_callback(self):

        try:
            for x, y in zip(self.data['X'], self.data['Y']):
                if self.i >= self.line_number:
                    self.get_logger().info('Process finished!')
                    self.destroy_node()
                    
                if x == 0.0 and y == 0.0:
                    continue

                else:
                    self.initial_point.point.x = x
                    self.initial_point.point.y = y
                    self.initial_point.point.z = 0.0
                    trans = self.buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time(seconds=0))
                    self.transformed_point = tf2_geometry_msgs.do_transform_point(self.initial_point, trans)
                    coords = [self.transformed_point.point.x, self.transformed_point.point.y]
                    self.save_csv(coords)
                    self.get_logger().info('{}: Transform: {}'.format(self.i, self.transformed_point))
                    self.i += 1

        except Exception as e:
            self.get_logger().info('Error')
            sys.exit()
    
    def line_count(self, path):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    line_count += 1

            return line_count
    
    def save_csv(self, coords):
        headerList = ["X axis", "Y axis"]
        filename = "/home/lascm/PycharmProjects/pythonProject/file2.csv"
        file_exists = os.path.isfile(filename)

        if coords == None:
            dx, dy = 0.0, 0.0
            with open("file2.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=';')
                if not file_exists:
                    writer.writerow(headerList)
                writer.writerow([dx, dy])
        else:
            dx, dy = coords
            with open("file2.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=';')
                if not file_exists:
                    writer.writerow(headerList)
                writer.writerow([dx, dy])
            

        

def main(args=None):

    rclpy.init(args=args)
    subscriber = Subscriber()
    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
