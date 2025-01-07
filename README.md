## Procedure of Camera Calibration

**Run the following script to solve for calibration using the data previously collected.**

```
python3 solve_calibration.py
```

**Run the following script to collect data.**
you need the ros2 nodes of robots running, and ros2 joystick node running. This script is an example, you need to change the topics accordingly

```
python3 collect_data.py
```

This script solves the correspondence between tag in robot base (world frame) and tag in camera frame. The result is the extrinsic matrix we need that transform pcd in the camera






