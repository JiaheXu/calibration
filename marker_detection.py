import cv2
import os, time
import numpy as np
import matplotlib.pyplot as plt


# original
def get_kinect_ir_frame(device, visualize=False):
    """
    Capture an IR frame from the Kinect camera.
    """
    # Capture an IR frame
    for _ in range(20):
        try:
            device.get_capture()
            capture = device.get_capture()
            if capture is not None:
                ir_frame = capture.ir
                ir_frame = np.clip(ir_frame, 0, 5e3) / 5e3  # Clip and normalize
                # cv2.imshow('IR', ir_frame)
                if visualize:
                    plt.imshow(ir_frame)
                    plt.show()
                return ir_frame
        except:
            time.sleep(0.1)
            print("Failed to capture IR frame.")
    else:
        print("Failed to capture IR frame after 20 attempts.")
        return None


# --
def get_kinect_rgb_frame(device, visualize=False):
    """
    Capture an IR frame from the Kinect camera.
    """
    # Capture an IR frame
    for _ in range(20):
        try:
            device.get_capture()
            capture = device.get_capture()
            if capture is not None:
                rgb_frame = capture.color
                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                gray_frame = np.clip(gray_frame, 0, 5e3) / 5e3  # Clip and normalize
                cv2.imshow('color', rgb_frame)
                if visualize:
                    plt.imshow(rgb_frame)
                    plt.show()
                return gray_frame
        except:
            time.sleep(0.1)
            print("Failed to capture IR frame.")
    else:
        print("Failed to capture IR frame after 20 attempts.")
        return None

    



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
