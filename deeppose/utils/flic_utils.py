import cv2
import numpy as np


flic_available_joints = [ 
    'L_Shoulder',
    'L_Elbow',
    'L_Wrist',
    'R_Shoulder',
    'R_Elbow',
    'R_Wrist',
    'L_Hip',
    'R_Hip',
    'L_Eye',
    'R_Eye',
    'Nose',
]

flic_joint_names = [
    # Body
    'L_Shoulder',
    'L_Elbow',
    'L_Wrist',
    'R_Shoulder',
    'R_Elbow',
    'R_Wrist',
    'L_Hip',
    'L_Knee',
    'L_Ankle',
    'R_Hip',
    'R_Knee',
    'R_Ankle',
    # Face
    'L_Eye',
    'R_Eye',
    'L_Ear',
    'R_Ear',
    'Nose',
    # ?
    'M_Shoulder',
    'M_Hip',
    'M_Ear',
    'M_Torso',
    'M_LUpperArm',
    'M_RUpperArm',
    'M_LLowerArm',
    'M_RLowerArm',
    'M_LUpperLeg',
    'M_RUpperLeg',
    'M_LLowerLeg',
    'M_RLowerLeg',
]

flic_joint_pairs = [
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (6, 0),
    (7, 3),
    (8, 10),
    (9, 10),
    (0, 3),
    (6, 7)
]


def draw_joints(img, point):
    img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.uint8)

    for start_i, end_i in flic_joint_pairs:
        st = tuple(int(v) for v in point[start_i, ::-1])
        en = tuple(int(v) for v in point[end_i, ::-1])
        cv2.line(img, st, en, (0, 0, 255), 2, cv2.LINE_AA)

    for y, x in point:
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 2, (255, 255, 255), -1, cv2.LINE_AA)
    
    return img[:, :, ::-1]
