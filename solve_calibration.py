import numpy as np
import cv2
import pickle, os
from scipy.linalg import logm, expm
from scipy.spatial.transform import Rotation

def solve_rigid_transformation(inpts, outpts):
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

    # Calculate the reprojection error
    avg_error = calculate_reprojection_error(tag_poses, target_poses_in_camera, T)
    print(f"Average reprojection error: {avg_error}")


    return T


if __name__ == "__main__":
    filepath = os.path.abspath(__file__)
    dirpath = os.path.dirname(filepath)

    # Load data
    cam_id = 1
    data_dirname = os.path.join(dirpath, "collected_data")
    data_filepath = os.path.join(data_dirname, f"cam{cam_id}_data.pkl")
    with open(data_filepath, "rb") as f:
        data = pickle.load(f)
    tag_poses, target_poses_in_camera = zip(*data) 
    
    # Solve the extrinsic calibration
    T = solve_extrinsic(tag_poses, target_poses_in_camera)

    # Save the calibration
    calib_dirname = os.path.join(dirpath, "calibration_results")
    os.makedirs(calib_dirname, exist_ok=True)
    filepath = os.path.join(calib_dirname, f"cam{cam_id}_calibration.npz")
    np.savez(filepath, T=T)
