import numpy as np
import cv2
import pickle, os
from scipy.linalg import logm, expm
from scipy.spatial.transform import Rotation

def solve_rigid_transformation(inpts, outpts):
    # R @ inpts + t = outpts
    # to_E_from
    # cam_R_base @ base_p + cam_T_base = CAM_p
    # inpts: base_p (points in base frame)
    # outpts: CAM_p (tag position in camera frame) 
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """
    assert inpts.shape == outpts.shape
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    outpts -= outpt_mean
    inpts -= inpt_mean
    X = inpts.T
    Y = outpts.T
    covariance = np.dot(X, Y.T)
    U, s, V = np.linalg.svd(covariance)
    S = np.diag(s)
    assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
    R = np.dot(np.dot(V, idmatrix), U.T)
    t = outpt_mean.T - np.dot(R, inpt_mean)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def calculate_reprojection_error(tag_poses, target_poses, T_matrix):
    errors = []
    idx = 0
    for tag_pose, target_pose in zip(tag_poses, target_poses):
        # Transform target pose using T_matrix
        # transformed_target = np.dot(T_matrix, target_pose)
        transformed_target = T_matrix@ target_pose
        # print("T_matrix.shape: ", T_matrix.shape)
        # print("target_pose.shape: ", target_pose.shape)
        transformed_pos = transformed_target[:3, 3]

        # Compare with tag pos
        tag_pos = tag_pose[:3, 3]
        error = np.linalg.norm(tag_pos - transformed_pos)
        print(idx, " error: ", error)
        idx += 1
        errors.append(error)

    # Compute average error
    # print("error: ", errors)
    avg_error = np.mean(errors)
    return avg_error

def solve_extrinsic(tag_poses, target_poses_in_camera):
    """
    Solve the extrinsic calibration between the camera and the base.
    """
    
    tag_pos = np.array([pose[:3, 3] for pose in tag_poses])
    target_pos = np.array([pose[:3, 3] for pose in target_poses_in_camera])
    T = solve_rigid_transformation(target_pos, tag_pos)
    print(f"Transformation matrix T:\n{T}")

    rot = Rotation.from_matrix( T[0:3, 0:3])
    quat = rot.as_quat(rot)
    trans = T[0:3, 3]
    print("quat: ", quat)
    print("trans: ", trans)
    # Calculate the reprojection error
    avg_error = calculate_reprojection_error(tag_poses, target_poses_in_camera, T)
    print(f"Average reprojection error: {avg_error}")


    return T

def get_matrix( transf ):
    T = np.eye(4)
    T[0][3] = transf[0]
    T[1][3] = transf[1]
    T[2][3] = transf[2]
    rot = Rotation.from_quat(transf[3:7])
    # print(rot.as_matrix())
    T[0:3,0:3] = rot.as_matrix()[0]
    return T


if __name__ == "__main__":


    data = np.load("./success.npy", allow_pickle = True)
    print("len: ", len(data))
    base_tags = []
    cam_tags = []


    for point in data:
        base_tag = point["base_tag"]
        base_tag_transform = get_matrix(base_tag)
        base_tags.append(base_tag_transform)
        # print("base: ", base_tag_transform)
        cam_tag = point["camera_tag"]
        cam_tag_transform = get_matrix(cam_tag)
        cam_tags.append(cam_tag_transform)
        # print("cam_tag_transform: ", cam_tag_transform)
        # print("")

    start = 50
    end = 200
    base_tags = base_tags[start : end]
    cam_tags = cam_tags[start : end]
    solve_extrinsic(base_tags, cam_tags)
