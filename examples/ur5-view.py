from __future__ import absolute_import, division, print_function

import math
import time
import os
import numpy as np
import meshcat
import random
import meshcat.geometry as g
import meshcat.transformations as tf

# Constants for UR5 arm dimensions
SHOULDER_OFFSET = 0.13585  # measured from model
ELBOW_OFFSET = 0.0165  # measured from model

# DH parameters of a UR5 arm
A1 = 0.425
A2 = 0.39225
D0 = 0.089159
D3 = 0.10915
D4 = 0.09465
D5 = 0.0823

def initialize_visualizer():
    """Initialize the MeshCat visualizer."""
    return meshcat.Visualizer().open()

def load_robot_parts(visualizer, parts):
    """Load the robot parts into the visualizer with random colors."""
    for part in parts:
        color = random.randint(0, 0xFFFFFF)
        visualizer[f"ur5/{part}"].set_object(
            g.DaeMeshGeometry.from_file(
                os.path.join(f"./Universal_Robots_ROS2_Description/meshes/ur5/visual/{part}.dae")),
            g.MeshLambertMaterial(color=color)
        )

def compute_transformation_matrix(joint_angles):
    """
    Compute the transformation matrix for each joint of the UR5 arm.

    Args:
        joint_angles (list): List of joint angles.

    Returns:
        list: List of transformation matrices for each joint.
    """
    T0 = tf.translation_matrix([0, 0, 0]).dot(tf.rotation_matrix(0, [0, 0, 1]))

    T1 = T0 @ tf.translation_matrix([0, 0, D0]).dot(tf.rotation_matrix(joint_angles[0], [0, 0, 1]))

    T2 = T1 @ tf.translation_matrix([0, SHOULDER_OFFSET, 0]).dot(
        tf.rotation_matrix(np.pi / 2, [0, 1, 0])).dot(tf.rotation_matrix(joint_angles[1], [0, 1, 0]))

    T3 = T2 @ tf.translation_matrix([0, -SHOULDER_OFFSET + ELBOW_OFFSET, A1]).dot(
        tf.rotation_matrix(joint_angles[2], [0, 1, 0]))

    T4 = T3 @ tf.translation_matrix([0, 0, A2]).dot(
        tf.rotation_matrix(np.pi / 2, [0, 1, 0])).dot(tf.rotation_matrix(joint_angles[3], [0, 1, 0]))

    T5 = T4 @ tf.translation_matrix([0, D3 - ELBOW_OFFSET, 0]).dot(tf.rotation_matrix(joint_angles[4], [0, 0, 1]))

    T6 = T5 @ tf.translation_matrix([0, 0, D4]).dot(tf.rotation_matrix(joint_angles[5], [0, 1, 0]))

    return [T0, T1, T2, T3, T4, T5, T6]

def create_capsule(visualizer, point1, point2, radius, color=0xFFFFFF, opacity=0.5):
    """
    Create a capsule shape between two points with the specified radius, color, and opacity.

    Args:
        visualizer (meshcat.Visualizer): The MeshCat visualizer instance.
        point1 (list or tuple): The first point [x, y, z].
        point2 (list or tuple): The second point [x, y, z].
        radius (float): The radius of the capsule.
        color (int): The color of the capsule in hexadecimal.
        opacity (float): The opacity of the capsule.
    """
    material = g.MeshLambertMaterial(color=color, opacity=opacity, transparent=True)
    
    # Create spheres at the endpoints
    visualizer["capsule/sphere1"].set_object(g.Sphere(radius), material)
    visualizer["capsule/sphere1"].set_transform(tf.translation_matrix(point1))
    
    visualizer["capsule/sphere2"].set_object(g.Sphere(radius), material)
    visualizer["capsule/sphere2"].set_transform(tf.translation_matrix(point2))
    
    # Create cylinder between the endpoints
    midpoint = [(p1 + p2) / 2 for p1, p2 in zip(point1, point2)]
    height = np.linalg.norm(np.array(point2) - np.array(point1))
    direction = np.array(point2) - np.array(point1)
    direction = direction / np.linalg.norm(direction)
    axis = np.cross([0, 1, 0], direction)
    angle = np.arccos(np.dot([0, 1, 0], direction))
    rotation_matrix = tf.rotation_matrix(angle, axis)
    
    visualizer["capsule/cylinder"].set_object(g.Cylinder(height, radius), material)
    visualizer["capsule/cylinder"].set_transform(tf.translation_matrix(midpoint).dot(rotation_matrix))


def main():
    """Main function to visualize the UR5 robot arm."""
    vis = initialize_visualizer()
    parts = ["base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3"]
    load_robot_parts(vis, parts)

    create_capsule(vis, [0, 0.1, 0], [0, -0.1, 0.5], 0.1)

    while True:
        t = time.time()
        joint_angles = [0.5 * math.sin(t + i) for i in range(6)]
        
        # Compute transformation matrices
        transforms = compute_transformation_matrix(joint_angles)

        # Set transformations for each part
        for i, part in enumerate(parts):
            vis[f"ur5/{part}"].set_transform(transforms[i])

        time.sleep(0.01)

if __name__ == "__main__":
    main()