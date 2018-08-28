#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.pose = None
        self.stopline_wp_idx = -1
        self.base_waypoints = None
        self.waypoints_2d = None
	self.waypoint_kdtree = None

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.waypoint_kdtree:
                # publish closest points
                self.publish_waypoints()
            rate.sleep()

    def publish_waypoints(self):
        self.final_waypoints_pub.publish(self.generate_lane())

    def generate_lane(self):
        lane = Lane()
        closest_wp_idx = self.get_closest_waypoint_idx()
        furthest_wp_idx = closest_wp_idx+LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_wp_idx:furthest_wp_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= furthest_wp_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.deccelerate_waypoints(base_waypoints, closest_wp_idx)
        return lane

    def deccelerate_waypoints(self, base_waypoints, closest_wp_idx):
        final_waypoints = []
        stop_idx = self.stopline_wp_idx - closest_wp_idx - 2 # stopping at 2 wp before the stop line
        for i, wp in enumerate(base_waypoints):
            p = Waypoint()
            p.pose = wp.pose
            if i >= stop_idx:
                vel = 0
            else:
                dist = self.distance(waypoints, i, stop_idx)
                vel = math.sqrt(2*MAX_DECEL*dist)
                if vel < 1: vel = 0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x) # speed limit
            final_waypoints.append(p)
        return final_waypoints

    def get_closest_waypoint_idx(self):
        position = self.pose.pose.position
        x, y = position.x, position.y
        closest_idx = self.waypoint_kdtree.query((x, y), k=1)[1] # get index

        closest_wp = self.waypoints_2d[closest_idx]
        prev_wp = self.waypoints_2d[closest_idx-1]

        # determine if the closest wp is in front of or behind the car
        cl_vect = np.array(closest_wp)
        prev_vect = np.array(prev_wp)
        pos_vect = np.array([x, y])

        front = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if front > 0:
            closest_idx = (closest_idx+1) % len(self.waypoints_2d)
        return closest_idx

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
       	self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_kdtree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
