import numpy as np


def calc_pcp(pred, target, parts):
    detected = []
    for start_i, end_i in parts:
        correct_len = np.sqrt((target[start_i] - target[end_i]) ** 2)
        correct_area_radius = correct_len / 2

        start_dis = np.sqrt((target[start_i] - pred[start_i]) ** 2)
        end_dis = np.sqrt((target[end_i] - pred[end_i]) ** 2)

        if start_dis <= correct_area_radius and end_dis <= correct_area_radius:
            detected.append(True)
        else:
            detected.append(False)

    return np.asarray(detected)


def calc_pdj(pred, target, parts):
    

