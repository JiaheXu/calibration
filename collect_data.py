"""
Uses Deoxys to control the robot and collect data for calibration.
"""
import numpy as np
import math
import os, pickle
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from pyk4a import PyK4A
from pyk4a.calibration import CalibrationType

from hacman_real_env.robot_controller import FrankaOSCController
from marker_detection import get_kinect_ir_frame, detect_aruco_markers, estimate_transformation, get_kinect_rgb_frame

import time

# --
# ros stuff
import message_filters
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
bridge = CvBridge()

def callback(data): 
      
    # print the actual message in its raw format 
    # rospy.loginfo("Here's what was subscribed: %s", data.data) 
      
    # otherwise simply print a convenient message on the terminal 
    print('Data from /cam1/rgb/image_raw received', type(data.data)) 
    image1 = bridge.imgmsg_to_cv2(data)
    

# --

USE_DEPTH = False

def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]

def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


# change this to a ROS subscriber node
def move_robot_and_record_data(
        cam_id, 
        num_movements=3, 
        debug=False,
        initial_joint_positions=None,
        initial_pos_rot = None):
    


    """
    Move the robot to random poses and record the necessary data.
    """
    
    # Initialize the robot
    robot = FrankaOSCController()

    # def callback(self, rgb_msg)
    #     self.rgb = rgb_msg # convert from ros msg to cv2

    # change to self.rgb
    # Initialize the camera

    k4a = PyK4A(device_id=cam_id)
    k4a.start()

    # rospy.Subscriber("/cam1/rgb/image_raw", Image, callback)
    


    if USE_DEPTH:
        camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.DEPTH)
        dist_coeffs = k4a.calibration.get_distortion_coefficients(CalibrationType.DEPTH)
    else:
        # --
        # camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
        # dist_coeffs = k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)
        # camera_matrix = np.array([[613.95083883,   0.0,         636.26091649],
        # [  0.0,         613.82102393, 360.91265066],
        # [  0.0,           0.0,           1.0        ]])
        # dist_coeffs = np.array([0.09878901, -0.05604852, -0.001523,   -0.00083489, 0.00505589])
        camera_matrix = np.array([[613.32427146,  0.,        633.94909346],
        [ 0.,        614.36077155, 363.33858573],
        [ 0.,          0.,          1.       ]])
        dist_coeffs = np.array([[ 0.09547761, -0.06461896, -0.00039569, -0.00243461, 0.02172413]])
        # --

    # R0, T0 = initial_pos_rot[0], initial_pos_rot[1]
    # quat0 = mat2quat(R0)
    # axis_angle0 = quat2axisangle(quat0)
    # robot.move_to(T0, use_rot = True, target_rot = R0, duration = 4)
    data = []
    for i in range(len(initial_joint_positions)):
        this_initial_joint_positions = initial_joint_positions[i]
        for _ in tqdm(range(num_movements)):
            print("point ", i)
            # Generate a random target delta pose
            random_delta_pos = [np.random.uniform(-0.1, 0.1, size=(3,))]
            print(f"\nRecorded {len(data)} data points.")
            # Generate a random target delta pose
            random_delta_pos = [np.random.uniform(-0.1, 0.1, size=(3,))]
            random_delta_axis_angle = [np.random.uniform(-0.6, 0.6, size=(3,))]
            
            robot.reset(joint_positions=this_initial_joint_positions)
            robot.move_by(random_delta_pos, random_delta_axis_angle, duration = 8)
            
            # T = T0 + random_delta_pos
            # axis_angle = axis_angle0 + random_delta_axis_angle
            # quat = axisangle2quat(axis_angle)
            # robot.move_to(np.array(T), target_quat = quat, duration = 4)

            # Get current pose of the robot 
            time.sleep(2)
            gripper_pose = robot.eef_pose
            print(f"Gripper pos: {gripper_pose[:3, 3]}")
            ir_frame = None

            # Capture IR frame from Kinect
            if USE_DEPTH:
                ir_frame = get_kinect_ir_frame(k4a)
            else:
                ir_frame = get_kinect_rgb_frame(k4a)

            if ir_frame is not None:
                # Detect ArUco markers and get visualization
                corners, ids = detect_aruco_markers(ir_frame, debug=debug)

                # Estimate transformation if marker is detected
                if ids is not None and len(ids) > 0:
                    transform_matrix = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
                    if transform_matrix is not None:
                        data.append((
                            gripper_pose,       # gripper pose in base
                            transform_matrix    # tag pose in camera
                        ))
            else:
                print("\033[91m" + "No IR frame captured." + "\033[0m")
    

    print(f"Recorded {len(data)} data points.")
    # Save data
    os.makedirs("hacman_real_env/pcd_obs_env/calibration/data", exist_ok=True)
    filepath = f"hacman_real_env/pcd_obs_env/calibration/data/cam{cam_id}_data.pkl"
    with open(f"hacman_real_env/pcd_obs_env/calibration/data/cam{cam_id}_data.pkl", "wb") as f:
        pickle.dump(data, f)
    return filepath


cam1_joint_positions = [
        [-0.55437239, -0.05180611,  0.95524054, -2.36760072, -0.58737754, 2.10306058, 2.86483819],
        [-0.69060415, -0.00311314,  0.73452985, -2.19796134, -0.62743314,  2.12473973, 2.84853125],
        [-1.83816947, -0.5108411,   2.15390625, -2.20161035, -0.09434892,  2.32223756, 2.77909085],
        # [-1.83935739, -0.32222198,  1.6626642,  -2.30280593, -0.37450542,  2.69886776,  2.6559697 ],
        [-1.57567918, -0.14877912,  1.56618407, -2.22918519, -0.34223448,  2.24174083,  2.59340441]
        # [-1.70241174, -0.2351264,   2.09751665, -1.80738419, -0.10691031,  1.79956142, 2.82880695],
        # [-1.87458021, -0.64988864,  1.93651241, -2.54632479, -0.46713021,  2.69897605, 2.86592678]
        # [-0.90664935, -0.3101612,   1.26099689, -2.92967691, -0.2726125,   2.41813255, 2.33732528]
    ]



cam1_matrix = np.array([[739.1656494140625, 0.0, 951.3283081054688],
                       [0.0, 739.1656494140625, 566.4527587890625],
                       [0.0, 0.0, 1.0]])


class Camera():
    def __init__(self, 
                 cam_id = 1, 
                 cam_topic = "/cam1/zed_node_A/left/image_rect_color",
                 cam_matrix = cam1_matrix,
                 num_movements=5
                 ):
        self.cam_id = cam_id
        self.num_movements = num_movements
        self.robot = FrankaOSCController()

        # self.camera_matrix = np.array([[613.32427146,  0.,        633.94909346],
        # [ 0.,        614.36077155, 363.33858573],
        # [ 0.,          0.,          1.       ]])
        # self.dist_coeffs = np.array([[ 0.09547761, -0.06461896, -0.00039569, -0.00243461, 0.02172413]])

        self.camera_matrix = cam_matrix
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # recorded data
        self.data = []
        self.joint_data = []

        # self.cam_topic = "/cam" + str(cam_id) + "/rgb/image_raw"
        self.cam_topic = cam_topic

        self.current_frame = None

    def callback(self, data): 
        print("call")
        self.current_frame = bridge.imgmsg_to_cv2(data)

    def move_robot_and_record(self, initial_joint_positions):
        # rospy.Subscriber("/cam" + str(self.cam_id) + "/rgb/image_raw", Image, self.callback)

        rospy.Subscriber(self.cam_topic, Image, self.callback)

        # Load the predefined dictionary
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        for i in range(len(initial_joint_positions)):
            this_initial_joint_positions = initial_joint_positions[i]
            for j in tqdm(range(self.num_movements)):
                print("point ", i)

                # Generate a random target delta pose
                random_delta_pos = [np.random.uniform(-0.12, 0.12, size=(3,))]
                random_delta_axis_angle = [np.random.uniform(-0.4, 0.4, size=(3,))]
                
                # move back to reset point then move to a new random point
                self.robot.reset(joint_positions=this_initial_joint_positions)
                self.robot.move_by(random_delta_pos, random_delta_axis_angle, duration = 8)

                time.sleep(2)
                gripper_pose = self.robot.eef_pose
                print(f"Gripper pos: {gripper_pose[:3, 3]}")

                if self.current_frame is not None:
                    gray_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                    gray_frame = np.clip(gray_frame, 0, 5e3) / 5e3  # Clip and normalize
                    # cv2.imshow('color', self.current_frame)
                    # time.sleep(4)

                    ir_frame = gray_frame
                    gray = cv2.convertScaleAbs(ir_frame, alpha=(255.0/ir_frame.max()))
                    # gray = cv2.convertScaleAbs(ir_frame)
        
                    # Detect markers
                    corners, ids, rejected = detector.detectMarkers(gray)   # top-left, top-right, bottom-right, and bottom-left corners

                    if ids is not None and len(ids) > 0:

                        # Visualize markers
                        vis_image = cv2.aruco.drawDetectedMarkers(gray.copy(), corners, ids)
                        cv2.imwrite("vis/vis_cam" + str(self.cam_id) + "_" + str(i) + "_" + str(j) + ".png",vis_image)
                        # cv2.destroyAllWindows()
                        # cv2.imshow('ArUco Marker Detection', vis_image)

                        joint_positions = self.robot.joint_positions
                        self.joint_data.append(joint_positions)

                        transform_matrix = estimate_transformation(corners, ids, self.camera_matrix, self.dist_coeffs)

                        if transform_matrix is not None:
                            self.data.append((
                                gripper_pose,       # gripper pose in base
                                transform_matrix    # tag pose in camera
                            ))

                    self.current_frame = None

                    print(f"\nRecorded {len(self.data)} data points.")

        print(f"Recorded {len(self.data)} data points.")
        # Save data
        os.makedirs("collected_data", exist_ok=True)
        filepath = f"collected_data/cam{self.cam_id}_data.pkl"
        with open(f"collected_data/cam{self.cam_id}_data.pkl", "wb") as f:
            pickle.dump(self.data, f)

        self.joint_data = np.array(self.joint_data)
        np.savetxt(f"collected_data/cam{self.cam_id}_joint_data.txt", self.joint_data, newline="\n\n")



def main():
    # cam_id = 2
    # # 0: right -     000003493812
    # # 2: left -     000880595012
    # # 1: front -    000180921812
    # # 3: back -     000263392612

    # # --
    # # 1: front 000184925212
    # # 2: left 000196925212
    # # 0: back 000259921812
    # # --

    # initial_joint_positions = {
    #     0: [-0.70556419, 0.33820318, 0.29427356, -2.05766904, 0.56290124, 1.89085646, -1.58229465],
    #     # 1: [-0.68299696, 0.65603606, 0.07339937, -1.45441668, -0.06963243, 2.11292397, 1.73479704],
    #     # 1: [-0.85532137, -0.26198281,  0.84026816, -2.73738181, -0.29243987,  2.66221224,  2.35233091],
    #     1: [-1.45415599, -0.78135363,  1.70612916, -2.35355173,  0.14741078,  2.35525889, 2.52100375],
    #     2: [-0.85532137, -0.26198281,  0.84026816, -2.73738181, -0.29243987,  2.66221224,  2.35233091],
    #     3: [-0.57346419, 0.39241199, 0.04834748, -2.25460585, 0.61730919, 3.71824636, 1.5602955]

    # }[cam_id]
    
    # Perform the movements and record data
    '''
    move_robot_and_record_data(
        cam_id=cam_id, num_movements=50, debug=False, 
        initial_joint_positions=initial_joint_positions)
    '''

    # --
    moves_per_pt = 15
    cam1_joint_positions = [
        # [-0.55437239, -0.05180611,  0.95524054, -2.36760072, -0.58737754, 2.10306058, 2.86483819],
        # [-0.69060415, -0.00311314,  0.73452985, -2.19796134, -0.62743314,  2.12473973, 2.84853125],
        # [-1.83816947, -0.5108411,   2.15390625, -2.20161035, -0.09434892,  2.32223756, 2.77909085],
        # [-1.57567918, -0.14877912,  1.56618407, -2.22918519, -0.34223448,  2.24174083,  2.59340441],
        # [-2.3859332,  -0.86357368,  2.1533203,  -2.01295581,  0.72143359,  2.66985355,  1.89886398], #
        # [-1.49155543, -0.12654208,  1.57892412, -2.44720267, -0.90682357,  2.22838793,  2.81830426],
        # [-2.35550098, -0.38105734,  1.94369907, -2.37309269,  0.36355023,  2.54158534,  1.70757139],
        # [-0.09680262,  0.31373923, -0.592957,   -2.19007649,  0.17658745,  2.4745553,  1.41781171],
        # [-0.48897135,  0.21060299,  0.44908941, -2.32505043, -0.1986648,   2.56021555,  2.45454782],

        [-0.3095683,   0.53386543, -0.27563208, -1.66965653,  0.06494098,  2.2376618,  2.02635383],
        [-0.36171402,  0.47076712, -0.30784553, -1.88466698,  0.06911615,  2.4766335,  1.80341822],
        [ 0.00494685,  0.60437024, -0.60003229, -2.03648432, -0.03687939,  2.69390816,  1.78568895],
        [ 0.23622115,  0.69353994, -0.89618729, -1.95924878,  0.7955479,   2.47342381,  1.17657859],
        # [ 0.08550101,  0.74835914, -0.64377919, -1.71837505,  0.78799076,  2.4059501,  1.50559559],
        # [ 0.57074488,  0.44179658, -0.63363224, -2.34272113,  0.52261622,  2.59522057,  1.96359469],
        # [ 0.01847659,  0.74573345, -0.45376881, -1.61082106,  0.19321952,  2.29952219,  2.34740633],
        # [-0.0481589,   0.31121922, -0.33639577, -2.254645,    0.19737888,  2.53075349,  1.92891333],

        # [-0.27456893,  0.27048565, -0.34662997, -2.20299452,  0.19486937,  2.53001216,  1.60178725],
        # [-2.73265313, -0.53442667,  2.29382893, -2.23384255,  0.47698388,  2.54650537,  1.62779169], #
        # [-2.79382525, -0.56461507,  2.19789854, -2.18864312,  0.6015327,   2.6927462,  1.27651441], #
        # [-1.45745414,  0.09985845,  1.45956092, -2.35356975, -0.81404507,  1.85355657,  2.6339051 ]
    ]
    # move_robot_and_record_data(
    #     cam_id=0, num_movements=moves_per_pt, debug=False, 
    #     initial_joint_positions=cam1_joint_positions
    # )
    my_camera1 = Camera(
        cam_id = 1, 
        cam_topic = "/cam1/zed_node_A/left/image_rect_color",
        cam_matrix = cam1_matrix,
        num_movements=20
    )
    my_camera1.move_robot_and_record(cam1_joint_positions)
    # --
    

if __name__ == "__main__":
    main()


    # example: https://github.com/JiaheXu/IGEV/blob/main/IGEV-Stereo/demo_ros.py