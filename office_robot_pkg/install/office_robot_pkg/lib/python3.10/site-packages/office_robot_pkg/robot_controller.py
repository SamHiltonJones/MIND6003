############## IMPORT LIBRARIES #################
# Python math library
import math 
 
# ROS client library for Python
import rclpy 
 
# Enables pauses in the execution of code
from time import sleep 
 
# Used to create nodes
from rclpy.node import Node
 
# Enables the use of the string message type
from std_msgs.msg import String 
 
# Twist is linear and angular velocity
from geometry_msgs.msg import Twist     
                     
# Handles LaserScan messages to sense distance to obstacles (i.e. walls)        
from sensor_msgs.msg import LaserScan    

from sensor_msgs.msg import Image    
 
# Handle Pose messages
from geometry_msgs.msg import Pose 
 
# Handle float64 arrays
from std_msgs.msg import Float64MultiArray
                     
# Handles quality of service for LaserScan data
from rclpy.qos import qos_profile_sensor_data 
 
# Scientific computing library
import numpy as np 

import cv2
from cv_bridge import CvBridge
 
class Controller(Node):
  """
  Create a Controller class, which is a subclass of the Node 
  class for ROS2.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    ##################### ROS SETUP ####################################################
    # Initiate the Node class's constructor and give it a name
    super().__init__('Controller')
 
    # Create a subscriber
    # This node subscribes to messages of type Float64MultiArray  
    # over a topic named: /state_est
    # The message represents the current estimated state:
    #   [x, y, yaw]
    # The callback function is called as soon as a message 
    # is received.
    # The maximum number of queued messages is 10.
    self.subscription = self.create_subscription(
                        Float64MultiArray,
                        '/state_est',
                        self.state_estimate_callback,
                        10)
    self.subscription  # prevent unused variable warning

    # Create a subscriber
    # This node subscribes to messages of type 
    # sensor_msgs/msg/Image   
    self.cam_subscriber = self.create_subscription(
                           Image,
                           '/depth_camera/image_raw',
                           self.image_callback,
                           10)
                            
    # Create a publisher
    # This node publishes the desired linear and angular velocity of the robot (in the
    # robot base_link coordinate frame) to the /cmd_vel topic. Using the diff_drive
    # plugin enables the robot to read this /cmd_vel topic and execute
    # the motion accordingly.
    self.publisher_ = self.create_publisher(
                      Twist, 
                      '/cmd_vel', 
                      10)
 
    # Initialize the LaserScan sensor readings to some large value
    # Values are in meters.
    self.left_dist = 999999.9 # Left
    self.leftfront_dist = 999999.9 # Left-front
    self.front_dist = 999999.9 # Front
    self.rightfront_dist = 999999.9 # Right-front
    self.right_dist = 999999.9 # Right
 
    ################### ROBOT CONTROL PARAMETERS ##################
    # Maximum forward speed of the robot in meters per second
    # Any faster than this and the robot risks falling over.
    self.forward_speed = 0.025
 
    # Current position and orientation of the robot in the global 
    # reference frame
    self.current_x = 0.0
    self.current_y = 0.0
    self.current_yaw = 0.0
 
    ############# WALL FOLLOWING PARAMETERS #######################     
    # Finite states for the wall following mode
    #  "turn left": Robot turns towards the left
    #  "search for wall": Robot tries to locate the wall        
    #  "follow wall": Robot moves parallel to the wall
    self.wall_following_state = "turn left"
         
    # Set turning speeds (to the left) in rad/s 
    # These values were determined by trial and error.
    self.turning_speed_wf_fast = 3.0  # Fast turn
    self.turning_speed_wf_slow = 0.05 # Slow turn
 
    # Wall following distance threshold.
    # We want to try to keep within this distance from the wall.
    self.dist_thresh_wf = 0.50 # in meters  
 
    # We don't want to get too close to the wall though.
    self.dist_too_close_to_wall = 0.19 # in meters
 
  def state_estimate_callback(self, msg):
    """
    Extract the position and orientation data. 
    This callback is called each time
    a new message is received on the '/state_est' topic
    """
    # Update the current estimated state in the global reference frame
    curr_state = msg.data
    self.current_x = curr_state[0]
    self.current_y = curr_state[1]
    self.current_yaw = curr_state[2]
 
    # Command the robot to keep following the wall      
    # self.follow_wall()


  def image_callback(self, msg):
    """
    This method gets called every time an Image message is 
    received on the '/depth_camera/image_raw' topic 
    """    

    #Can uncomment to view image, does not work atm

    # try:
    #   # Convert ROS Image message to OpenCV image
    #   cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    #   # Display the image (you can modify this part as needed)
    #   cv2.imshow("Camera Image", cv_image)
    #   cv2.waitKey(1)  # Necessary to update image display
    # except Exception as e:
    #     print(e)

    #Test to see if I get image
    # print("image")
  
 
def main(args=None):
 
    # Initialize rclpy library
    rclpy.init(args=args)
     
    # Create the node
    controller = Controller()
 
    # Spin the node so the callback function is called
    # Pull messages from any topics this node is subscribed to
    # Publish any pending messages to the topics
    rclpy.spin(controller)
 
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    controller.destroy_node()
     
    # Shutdown the ROS client library for Python
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()
