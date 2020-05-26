import numpy as np
import cv2
from flowlib import flow_to_image

def visualize_score(score_map, big_number):
    lower_bound = -1 * big_number
    upper_bound = big_number
    all_values = np.reshape(score_map, (-1, ))
    all_values = all_values[all_values > lower_bound]
    all_values = all_values[all_values < upper_bound]
    max_val = all_values.max()
    min_val = all_values.min()
    visual_map = (score_map - min_val) / (max_val - min_val)
    visual_map[score_map == lower_bound] = 0
    visual_map[score_map == upper_bound] = 1
    visual_map *= 255
    visual_map = visual_map.astype(np.uint8)
    return visual_map

def visualize_img(img):
    if img.shape[2] == 2:
        cv2.imshow('Optical flow', flow_to_image(img))
    else:
        cv2.imshow('Image', img)
    cv2.waitKey(0)

def visualize_batch(batch):
    if len(batch.shape) == 4:
        if batch.shape[3] == 2:
            batch = [flow_to_image(batch[i]) for i in range(batch.shape[0])]
            cv2.imshow('Optical flow set', np.hstack(batch))
        else:
            batch = [batch[i] for i in range(batch.shape[0])]
            cv2.imshow('Image sets', np.hstack(batch))
        cv2.waitKey(0)
    else:
        if batch.shape[4] == 2:
            batch = [np.hstack([flow_to_image(batch[j][i]) for i in range(batch[j].shape[0])]) for j in range(batch.shape[0])]
            cv2.imshow('Optical flow set', np.vstack(batch))
        else:
            batch = [np.hstack([batch[j][i] for i in range(batch[j].shape[0])]) for j in range(batch.shape[0])]
            cv2.imshow('Image sets', np.vstack(batch))
        cv2.waitKey(0)

def visualize_pair(batch_1, batch_2):
    if len(batch_1.shape) == 4:
        if batch_1.shape[3] == 2:
            batch_1 = [flow_to_image(batch_1[i]) for i in range(batch_1.shape[0])]
        else:
            batch_1 = [batch_1[i] for i in range(batch_1.shape[0])]
        if batch_2.shape[3] == 2:
            batch_2 = [flow_to_image(batch_2[i]) for i in range(batch_2.shape[0])]
        else:
            batch_2 = [batch_2[i] for i in range(batch_2.shape[0])]
        cv2.imshow('Pair comparison', np.vstack([np.hstack(batch_1), np.hstack(batch_2)]))
        cv2.waitKey(0)
    else:
        if batch_1.shape[4] == 2:
            batch_1 = [flow_to_image(batch_1[-1][i]) for i in range(batch_1[-1].shape[0])]
        else:
            batch_1 = [batch_1[-1][i] for i in range(batch_1[-1].shape[0])]
        if batch_2.shape[4] == 2:
            batch_2 = [flow_to_image(batch_2[-1][i]) for i in range(batch_2[-1].shape[0])]
        else:
            batch_2 = [batch_2[-1][i] for i in range(batch_2[-1].shape[0])]
        cv2.imshow('Pair comparison', np.vstack([np.hstack(batch_1), np.hstack(batch_2)]))
        cv2.waitKey(0)
