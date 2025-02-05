"""
Uses Deoxys to control the robot and collect data for calibration.
"""
import numpy as np
import math
import os, pickle
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import time

# --
# ros stuff
import message_filters
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
bridge = CvBridge()
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
        self.robot = None
        # self.robot = FrankaOSCController() # replace to other controller

        self.camera_matrix = cam_matrix
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # recorded data
        self.data = []
        self.joint_data = []
        self.cam_topic = cam_topic
        self.current_frame = None

    def callback(self, data): 
        print("img callback")
        self.current_frame = bridge.imgmsg_to_cv2(data)

    def estimate_transformation(corners, ids, camera_matrix, dist_coeffs):
        """
        Estimate the transformation matrix A given ArUco marker detections.

        These should be known or calibrated beforehand:
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            dist_coeffs = np.array([k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4])  #  of 4, 5, 8 or 12 elements.
        """
        if ids is not None and len(ids) > 0:
            # Assuming marker size is known
            marker_size = 0.045  # In meters

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            # For demonstration, we'll use the first detected marker
            rvec, tvec = rvecs[0], tvecs[0]

            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            # Form the transformation matrix
            transform_mat = np.eye(4)
            transform_mat[:3, :3] = rmat
            transform_mat[:3, 3] = tvec.squeeze()
            return transform_mat

        return None

    def detect_aruco_markers(ir_frame, debug=False):
        """
        Detect ArUco markers in an IR frame and visualize the detection.
        """
        gray = cv2.convertScaleAbs(ir_frame, alpha=(255.0/ir_frame.max()))
        # gray = cv2.convertScaleAbs(ir_frame)
        
        # Load the predefined dictionary
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)   # top-left, top-right, bottom-right, and bottom-left corners

        # Visualize markers
        vis_image = cv2.aruco.drawDetectedMarkers(gray.copy(), corners, ids)
        # cv2.destroyAllWindows()
        cv2.imshow('ArUco Marker Detection', vis_image)
        cv2.waitKey(0 if debug else 1)

        return corners, ids

    def move_robot_and_record(self, initial_joint_positions):
        # rospy.Subscriber("/cam" + str(self.cam_id) + "/rgb/image_raw", Image, self.callback)

        rospy.Subscriber(self.cam_topic, Image, self.callback) # ??????

        # Load the predefined dictionary
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        for i in range(len(initial_joint_positions)):
            
            this_initial_joint_positions = initial_joint_positions[i] # get joint value

            for j in tqdm(range(self.num_movements)):
                print("point ", i)

                # Generate a random target delta pose
                random_delta_pos = [np.random.uniform(-0.12, 0.12, size=(3,))]
                random_delta_axis_angle = [np.random.uniform(-0.4, 0.4, size=(3,))]
                
                # move back to reset point then move to a new random point
                self.robot.reset(joint_positions=this_initial_joint_positions)
                self.robot.move_by(random_delta_pos, random_delta_axis_angle, duration = 8)

                time.sleep(2)
                #################################################################################################
                # get ee pose
                gripper_pose = self.robot.eef_pose
                #################################################################################################
                print(f"Gripper pos: {gripper_pose[:3, 3]}")


                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                # Detect markers
                corners, ids, rejected = detector.detectMarkers(gray)   # top-left, top-right, bottom-right, and bottom-left corners
                # corners are 2D lists

                
                # found a tag
                if ids is not None and len(ids) > 0:

                    # Visualize markers
                    vis_image = cv2.aruco.drawDetectedMarkers(gray.copy(), corners, ids)
                    cv2.imwrite("vis/vis_cam" + str(self.cam_id) + "_" + str(i) + "_" + str(j) + ".png",vis_image)
                    # cv2.destroyAllWindows()
                    # cv2.imshow('ArUco Marker Detection', vis_image)

                    transform_matrix = self.estimate_transformation(corners, ids, self.camera_matrix, self.dist_coeffs)

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

        # self.joint_data = np.array(self.joint_data)
        # np.savetxt(f"collected_data/cam{self.cam_id}_joint_data.txt", self.joint_data, newline="\n\n")



def main():

    # --
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
    
if __name__ == "__main__":
    main()