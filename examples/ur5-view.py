from __future__ import absolute_import, division, print_function

import math
import time
import os
import numpy as np

import meshcat
import random
import meshcat.geometry as g
import meshcat.transformations as tf


# https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/rolling/config/ur5

SHOULDER_OFFSET = 0.13585 # measured from model
ELBOW_OFFSET = 0.0165 # measured from model

# DH parameters of a UR5 arm
A1 = 0.425
A2 = 0.39225
D0 = 0.089159
D3 = 0.10915
D4 = 0.09465
D5 = 0.0823

vis = meshcat.Visualizer().open()

parts = ["base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3"]

for part in parts:
    color = random.randint(0, 0xFFFFFF)
    vis[f"ur5/{part}"].set_object(
        g.DaeMeshGeometry.from_file(
            os.path.join(f"./Universal_Robots_ROS2_Description/meshes/ur5/visual/{part}.dae")),
        g.MeshLambertMaterial(color=color)
    )

# Function to compute transformation matrix for each joint
def compute_transformation_matrix(joint_angles):
    T0 = tf.translation_matrix([0, 0, 0]).dot(
        tf.rotation_matrix(0, [0, 0, 1]))

    T1 = T0 @ tf.translation_matrix([0, 0, D0]).dot(
        tf.rotation_matrix(joint_angles[0], [0, 0, 1]))

    T2 = T1 @ tf.translation_matrix([0, SHOULDER_OFFSET, 0]).dot(
        tf.rotation_matrix(np.pi/2, [0, 1, 0])).dot(
        tf.rotation_matrix(joint_angles[1], [0, 1, 0]))

    T3 = T2 @ tf.translation_matrix([0, -SHOULDER_OFFSET+ELBOW_OFFSET, A1]).dot(
        tf.rotation_matrix(joint_angles[2], [0, 1, 0]))

    T4 = T3 @ tf.translation_matrix([0, 0, A2]).dot(
        tf.rotation_matrix(np.pi/2, [0, 1, 0])).dot(
        tf.rotation_matrix(joint_angles[3], [0, 1, 0]))

    T5 = T4 @ tf.translation_matrix([0, D3-ELBOW_OFFSET, 0]).dot(
        tf.rotation_matrix(joint_angles[4], [0, 0, 1]))

    T6 = T5 @ tf.translation_matrix([0, 0, D4]).dot(
        tf.rotation_matrix(joint_angles[5], [0, 1, 0]))

    return [T0, T1, T2, T3, T4, T5, T6]

# Loop to update joint angles in a sine wave and apply transformations
while True:
    t = time.time()
    joint_angles = [0.5 * math.sin(t), 0.5 * math.sin(t + 1), 0.5 * math.sin(t + 2), 0.5 * math.sin(t + 3), 0.5 * math.sin(t + 4), 0.5 * math.sin(t + 5)]
    
    # Compute transformation matrices
    transforms = compute_transformation_matrix(joint_angles)

    # Set transformations for each part
    for i, part in enumerate(parts):
        vis[f"ur5/{part}"].set_transform(transforms[i])

    time.sleep(0.01)