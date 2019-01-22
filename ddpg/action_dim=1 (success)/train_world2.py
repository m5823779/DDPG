import gym
import rospy
import roslaunch
import numpy as np
import random
import math

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, GetModelStateRequest

map_x_length = 8
map_y_length = 8
diagonal_dis = math.sqrt((map_x_length)** 2 +(map_y_length)** 2)

class train_world2(gazebo_env.GazeboEnv):
    def __init__(self):
        gazebo_env.GazeboEnv.__init__(self, "train_world2.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.model_coord = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self._seed()

    def calculate_observation(self, data):
        # Get the information of robot
        model = GetModelStateRequest()
        model.model_name = 'mobile_base'

        # Check whether bump the wall or arrive the target
        min_range = 0.2
        arrive = False
        done = False

        # robot informance
        self.position = self.model_coord(model).pose.position
        robot_angle = self.model_coord(model).pose.orientation

        # robot angle
        q_x, q_y, q_z, q_w = robot_angle.x, robot_angle.y, robot_angle.z, robot_angle.w
        yaw = math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z)))
        if yaw >= 0:
            yaw = yaw
        else:
            yaw = yaw + 360

        # Calculate the relative distance
        rel_dis_x = self.init_pose.position.x - self.position.x
        rel_dis_y = self.init_pose.position.y - self.position.y
        rel_dis = math.sqrt((rel_dis_x) ** 2 + (rel_dis_y) ** 2)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1/2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3/2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = math.degrees(theta)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = diff_angle
        else:
            diff_angle = 360 - diff_angle

        # Check whether the robot bump the wall
        for i, find in enumerate(data.ranges[4: 14]):
            if min_range > find > 0:
                done = True

        # Check whether the robot arrive the target
        if rel_dis < 0.4:
            arrive = True
            done = True

        # Laser signal normalization
        if data.range_max > diagonal_dis:
            norm_para = diagonal_dis
        else:
            norm_para = data.range_max

        return np.array(data.ranges[4: 14], dtype=float) / norm_para, rel_dis, rel_theta, diff_angle, done, arrive

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # Enter the linear , angular velocity
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.5
        vel_cmd.angular.z = action
        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        laser, rel_dis, rel_theta, diff_angle, done, arrive = self.calculate_observation(data)
        # Integrate all state and do normalization
        state = np.hstack([laser, np.array([rel_dis / diagonal_dis, rel_theta / 360])])

        if arrive:
            reward = 20
        else:
            if not done:
                reward = 10 * (self.past_rel_dis - rel_dis)
            else:
                reward = -20

        if diff_angle <= 30:
            reward_orientation = 0.3 * (30 - diff_angle)
        else:
            reward_orientation = 0.03 * (30 - diff_angle)

        total_reward = reward + reward_orientation

        # In order to know whether robot closer to the target save the past relative distance
        self.past_rel_dis = rel_dis

        # print('reward: ', reward, '| reward_orientation:', reward_orientation)

        return state, total_reward, done, arrive

    def _reset(self):
        # Reset the env #
        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('target')
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Build the target
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            self.init_pose = Pose()
            goal_urdf = open("/home/airlab/gym-gazebo/gym_gazebo/envs/assets/models/Target/model.sdf", "r").read()
            target = SpawnModel
            target.model_name = 'target'  # the same with sdf name
            target.model_xml = goal_urdf
            self.init_pose.position.x = random.uniform(-3.2, 3.2)
            self.init_pose.position.y = random.uniform(-3.2, 3.2)
            self.goal(target.model_name, target.model_xml, 'namespace', self.init_pose, 'world')

        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        rospy.wait_for_service('/gazebo/set_model_state')

        # Read laser data #
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        laser, rel_dis, rel_theta, diff_angle, done, arrive = self.calculate_observation(data)
        state = np.hstack([laser, np.array([rel_dis / diagonal_dis, rel_theta / 360])])

        # Build past rel_dis
        self.past_rel_dis = rel_dis

        return state

