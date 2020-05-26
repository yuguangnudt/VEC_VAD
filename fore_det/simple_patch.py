import numpy as np
import itertools


def get_patch_loc(h, w, h_num, w_num):
    h_step = h / h_num
    w_step = w / w_num
    y_min_list = np.linspace(0, h-1, h_num, endpoint=False)
    x_min_list = np.linspace(0, w-1, w_num, endpoint=False)
    patch_loc = list()
    for x_min, y_min in itertools.product(tuple(x_min_list), tuple(y_min_list)):
        x_max = np.minimum(x_min + w_step, w-1)
        y_max = np.minimum(y_min + h_step, h-1)
        patch_loc.append(np.array([x_min, y_min, x_max, y_max]))
    patch_loc = np.array(patch_loc)
    return patch_loc

