#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml
import numpy as np
import cv2
from cv_bridge import CvBridge

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_kdtree = None
        self.camera_image = None
        self.lights = []
	self.img_id = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)	

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.process_light_pub = rospy.Publisher('/process_light', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.process_light_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_kdtree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def save_img_to_file(self, msg):
	bridge = CvBridge()
	image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
	cv2.imwrite('../../../data/test/test_'+str(self.img_id)+'.png', image)
	self.img_id += 1

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        if self.process_light_count < 15:
            self.process_light_count += 1
            light_wp, state = self.last_wp, self.state
        else:
            light_wp, state = self.process_traffic_lights()
            light_wp_code = light_wp if state == TrafficLight.RED else -1
            self.process_light_pub.publish(Int32(light_wp_code))
            self.process_light_count = 0
	# collect data to build classifier
        #self.save_img_to_file(msg)
	'''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_wp_idx = self.waypoint_kdtree.query([x, y], 1)[1] # get index
        return closest_wp_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #return light.state # for now, before we build the model
        if(not self.has_image):
            self.prev_light_loc = None
            return False
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        state = self.light_classifier.get_classification(cv_image)
        return state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            position = self.pose.pose.position
            car_wp_idx = self.get_closest_waypoint(position.x, position.y)
            minimum = len(self.waypoints.waypoints)
            # find closest light
            for i, l in enumerate(self.lights):
                stop_line = stop_line_positions[i]
                wp_closest_to_line_idx = self.get_closest_waypoint(stop_line[0], stop_line[1])
                distance = wp_closest_to_line_idx-car_wp_idx
                front = bool(distance>0)
                # if there is a light in front
                if front and distance < minimum:
                    minimum = distance
                    light = l
                    light_wp = wp_closest_to_line_idx
        if light:
            state = self.get_light_state(light)
            return light_wp, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
