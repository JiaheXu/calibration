# --
# ros stuff
import message_filters
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs.point_cloud2 import read_points
# import sensor_msgs_py
from cv_bridge import CvBridge
bridge = CvBridge()

from hacman_real_env.robot_controller import FrankaOSCController

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

import open3d as o3d


class Cloud():
    def __init__(
            self,
            topic = "/points_concatenated"
            # topic = "/cam3/zed_node_C/rgb/image_rect_color"
    ):
        self.topic = topic
        self.robot = FrankaOSCController()
        self.current_frame = None

        self.DUMMY_FIELD_PREFIX = '__'

        self.cloud_arr = None

    def callback(self, data): 
        # self.current_frame = bridge.imgmsg_to_cv2(data)
        print("there")
        gen = read_points(data)
        # print(gen)
        # breakpoint()
        res = []
        for pts in gen:
            # print(type(pts[0]), pts[0])
            # print(pts)
            # breakpoint()
            # if pts[0] == float("NaN") or pts[0] is None:
            #     breakpoint()
            # print(pts.shape)
            if math.isnan(pts[0]) is False:
            #     breakpoint()
            # if float("nan") not in pts and "nan" not in pts:
                res.append(list(pts))
        self.cloud_arr = np.array(res)
        # breakpoint()
        # self.current_frame = bridge.imgmsg_to_cv2(data)
        #  = self.pointcloud2_to_array(data)
        print("updated self.cloud arr")

    def fields_to_dtype(self, fields, point_step):
        '''Convert a list of PointFields to a numpy record datatype.
        '''
        
        type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]
        pftype_to_nptype = dict(type_mappings)
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(
                    ('%s%d' % (self.DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1

            dtype = pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += pftype_to_nptype[f.datatype].itemsize * f.count

        # might be extra padding between points
        while offset < point_step:
            np_dtype_list.append(('%s%d' % (self.DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        return np_dtype_list
    
    def pointcloud2_to_array(self, cloud_msg, squeeze=True):
        ''' Converts a rospy PointCloud2 message to a numpy recordarray

        Reshapes the returned array to have shape (height, width), even if the
        height is 1.

        The reason for using np.frombuffer rather than struct.unpack is
        speed... especially for large point clouds, this will be <much> faster.
        '''
        # construct a numpy record type equivalent to the point type of this cloud
        dtype_list = self.fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

        # parse the cloud into an array
        cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

        # remove the dummy fields that were added
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (
                fname[:len(self.DUMMY_FIELD_PREFIX)] == self.DUMMY_FIELD_PREFIX)]]

        if squeeze and cloud_msg.height == 1:
            return np.reshape(cloud_arr, (cloud_msg.width,))
        else:
            return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


    def get_full_pcd(self):
        print("hi")
        rospy.Subscriber(self.topic, PointCloud2, self.callback)
        time.sleep(2)
        if self.cloud_arr is not None:
            print("collected cloud arr", type(self.cloud_arr))
            print(self.cloud_arr)
        # for item in self.cloud_arr:
        #     print(item, type(item))
        pcd_arr = self.cloud_arr[:, :3]
        print(pcd_arr)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_arr)
        o3d.visualization.draw_geometries([pcd])

# class Cloud():
#     def __init__(self, 
#                  cam_id = 1, 
#                  cam_topic = "/cam1/zed_node_A/left/image_rect_color",
#                  cam_matrix = None,
#                  num_movements=5
#                  ):
#         self.cam_id = cam_id
#         self.num_movements = num_movements
#         self.robot = FrankaOSCController()

#         # self.camera_matrix = np.array([[613.32427146,  0.,        633.94909346],
#         # [ 0.,        614.36077155, 363.33858573],
#         # [ 0.,          0.,          1.       ]])
#         # self.dist_coeffs = np.array([[ 0.09547761, -0.06461896, -0.00039569, -0.00243461, 0.02172413]])

#         self.camera_matrix = cam_matrix
#         self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

#         # recorded data
#         self.data = []
#         self.joint_data = []

#         # self.cam_topic = "/cam" + str(cam_id) + "/rgb/image_raw"
#         self.cam_topic = cam_topic

#         self.current_frame = None

#     def callback(self, data): 
#         self.current_frame = bridge.imgmsg_to_cv2(data)

#     def get_full_pcd(self):
#         # rospy.Subscriber("/cam" + str(self.cam_id) + "/rgb/image_raw", Image, self.callback)

#         rospy.Subscriber(self.cam_topic, Image, self.callback)
#         time.sleep(20)

#         # Load the predefined dictionary
#         # dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
#         # parameters =  cv2.aruco.DetectorParameters()
#         # detector = cv2.aruco.ArucoDetector(dictionary, parameters)

#         # for i in range(len(initial_joint_positions)):
#         #     this_initial_joint_positions = initial_joint_positions[i]
#         #     for j in tqdm(range(self.num_movements)):
#         #         print("point ", i)

#         #         # Generate a random target delta pose
#         #         random_delta_pos = [np.random.uniform(-0.12, 0.12, size=(3,))]
#         #         random_delta_axis_angle = [np.random.uniform(-0.4, 0.4, size=(3,))]
                
#         #         # move back to reset point then move to a new random point
#         #         self.robot.reset(joint_positions=this_initial_joint_positions)
#         #         self.robot.move_by(random_delta_pos, random_delta_axis_angle, duration = 8)

#         #         time.sleep(2)
#         #         gripper_pose = self.robot.eef_pose
#         #         print(f"Gripper pos: {gripper_pose[:3, 3]}")

#         #         if self.current_frame is not None:
#         #             gray_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
#         #             gray_frame = np.clip(gray_frame, 0, 5e3) / 5e3  # Clip and normalize
#         #             # cv2.imshow('color', self.current_frame)
#         #             # time.sleep(4)

#         #             ir_frame = gray_frame
#         #             gray = cv2.convertScaleAbs(ir_frame, alpha=(255.0/ir_frame.max()))
#         #             # gray = cv2.convertScaleAbs(ir_frame)
        
#         #             # Detect markers
#         #             corners, ids, rejected = detector.detectMarkers(gray)   # top-left, top-right, bottom-right, and bottom-left corners

#         #             if ids is not None and len(ids) > 0:

#         #                 # Visualize markers
#         #                 vis_image = cv2.aruco.drawDetectedMarkers(gray.copy(), corners, ids)
#         #                 cv2.imwrite("vis/vis_cam" + str(self.cam_id) + "_" + str(i) + "_" + str(j) + ".png",vis_image)
#         #                 # cv2.destroyAllWindows()
#         #                 # cv2.imshow('ArUco Marker Detection', vis_image)

#         #                 joint_positions = self.robot.joint_positions
#         #                 self.joint_data.append(joint_positions)

#         #                 transform_matrix = estimate_transformation(corners, ids, self.camera_matrix, self.dist_coeffs)

#         #                 if transform_matrix is not None:
#         #                     self.data.append((
#         #                         gripper_pose,       # gripper pose in base
#         #                         transform_matrix    # tag pose in camera
#         #                     ))

#         #             self.current_frame = None

#         #             print(f"\nRecorded {len(self.data)} data points.")

#         # print(f"Recorded {len(self.data)} data points.")
#         # # Save data
#         # os.makedirs("collected_data", exist_ok=True)
#         # filepath = f"collected_data/cam{self.cam_id}_data.pkl"
#         # with open(f"collected_data/cam{self.cam_id}_data.pkl", "wb") as f:
#         #     pickle.dump(self.data, f)

#         # self.joint_data = np.array(self.joint_data)
#         # np.savetxt(f"collected_data/cam{self.cam_id}_joint_data.txt", self.joint_data, newline="\n\n")




if __name__ == "__main__":
    my_cloud = Cloud()
    my_cloud.get_full_pcd()