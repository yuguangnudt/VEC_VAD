import mmcv
from mmcv.image import imread, imwrite
import cv2
from fore_det.inference import inference_detector, init_detector, show_result
import numpy as np
from sklearn import preprocessing
import os
from torch.utils.data import Dataset, DataLoader
from vad_datasets import ped_dataset, avenue_dataset, shanghaiTech_dataset
from configparser import ConfigParser
import time

cp = ConfigParser()
cp.read("config.cfg")

def getObBboxes(img, model, dataset_name):
    if dataset_name == 'UCSDped2':
        score_thr = 0.5
        min_area_thr = 10 * 10
    elif dataset_name == 'avenue':
        score_thr = 0.25
        min_area_thr = 40 * 40
    elif dataset_name == 'ShanghaiTech':
        score_thr = 0.5
        min_area_thr = 8 * 8
    else:
        raise NotImplementedError
    
    result = inference_detector(model, img)
    
    #bboxes = show_result(img, result, model.CLASSES, score_thr)
    bbox_result = result
    bboxes = np.vstack(bbox_result)
    
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    bbox_areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    
    return bboxes[bbox_areas >= min_area_thr, :4]

def delCoverBboxes(bboxes, dataset_name):
    if dataset_name == 'UCSDped2':
        cover_thr = 0.6
    elif dataset_name == 'avenue':
        cover_thr = 0.6
    elif dataset_name == 'ShanghaiTech':
        cover_thr = 0.65
    else:
        raise NotImplementedError

    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 4
    
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    bbox_areas = (y2-y1+1) * (x2-x1+1)

    sort_idx = bbox_areas.argsort()#Index of bboxes sorted in ascending order by area size
    
    keep_idx = []
    for i in range(sort_idx.size):
        #Calculate the point coordinates of the intersection
        x11 = np.maximum(x1[sort_idx[i]], x1[sort_idx[i+1:]]) 
        y11 = np.maximum(y1[sort_idx[i]], y1[sort_idx[i+1:]])
        x22 = np.minimum(x2[sort_idx[i]], x2[sort_idx[i+1:]])
        y22 = np.minimum(y2[sort_idx[i]], y2[sort_idx[i+1:]])
        #Calculate the intersection area
        w = np.maximum(0, x22-x11+1)    
        h = np.maximum(0, y22-y11+1)  
        overlaps = w * h
        
        ratios = overlaps / bbox_areas[sort_idx[i]]
        num = ratios[ratios > cover_thr]
        if num.size == 0:  
            keep_idx.append(sort_idx[i])
            
    return bboxes[keep_idx]
        
        

def imshow_bboxes(img,
                  bboxes,
                  bbox_color=(255,255,255),
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4).
        bbox_color (RGB value): Color of bbox lines.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5


    for bbox in bboxes:
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
                                                            
        img = cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness)
        
    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        imwrite(img, out_file)

        
def getFgBboxes(cur_img, img_batch, bboxes, dataset_name, verbose=False):
    if dataset_name == 'UCSDped2':
        area_thr = 10 * 10
        binary_thr = 18
        extend = 2
        gauss_mask_size = 3
    elif dataset_name == 'avenue':
        area_thr = 40 * 40
        binary_thr = 18
        extend = 2
        gauss_mask_size = 5
    elif dataset_name == 'ShanghaiTech':
        area_thr = 8 * 8
        binary_thr = 15
        extend = 2
        gauss_mask_size = 5
    else:
        raise NotImplementedError

    sum_grad = 0
    for i in range(img_batch.shape[0]-1):
        img1 = img_batch[i,:,:,:]
        img2 = img_batch[i+1,:,:,:]
        img1 = cv2.GaussianBlur(img1, (gauss_mask_size, gauss_mask_size), 0)
        img2 = cv2.GaussianBlur(img2, (gauss_mask_size, gauss_mask_size), 0)
        
        grad = cv2.absdiff(img1, img2)
        sum_grad = grad + sum_grad

    sum_grad = cv2.threshold(sum_grad, binary_thr, 255, cv2.THRESH_BINARY)[1]
    if verbose is True:
        cv2.imshow('grad', sum_grad)
        cv2.waitKey(0)
    
    for bbox in bboxes:
        bbox_int = bbox.astype(np.int32)
        extend_y1 = np.maximum(0, bbox_int[1]-extend)
        extend_y2 = np.minimum(bbox_int[3]+extend, sum_grad.shape[0])
        extend_x1 = np.maximum(0, bbox_int[0]-extend)
        extend_x2 = np.minimum(bbox_int[2]+extend, sum_grad.shape[1])
        sum_grad[extend_y1:extend_y2+1, extend_x1:extend_x2+1] = 0
    if verbose is True:
        cv2.imshow('del_ob_bboxes', sum_grad)
        cv2.waitKey(0)

    sum_grad = cv2.cvtColor(sum_grad, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(sum_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_bboxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        sum_grad = cv2.rectangle(sum_grad, (x,y), (x+w,y+h), 255, 1)
        area = (w+1) * (h+1)
        if area > area_thr and w / h < 10 and h / w < 10:
            extend_x1 = np.maximum(0, x-extend)
            extend_y1 = np.maximum(0, y-extend)
            extend_x2 = np.minimum(x+w+extend, sum_grad.shape[1])
            extend_y2 = np.minimum(y+h+extend, sum_grad.shape[0])
            fg_bboxes.append([extend_x1, extend_y1, extend_x2, extend_y2])
            cur_img = cv2.rectangle(cur_img, (extend_x1,extend_y1), (extend_x2,extend_y2), (0,255,0), 1)

    if verbose is True:
        cv2.imshow('all_fg_bboxes', sum_grad)
        cv2.waitKey(0)
        cv2.imshow('filter', cur_img)
        cv2.waitKey(0)
    
    return np.array(fg_bboxes)
    
            
            
def getBatch(data_folder, X_dataset, context_frame_num, idx, mode, file_format):
    dataset = X_dataset(dir=data_folder, context_frame_num=context_frame_num, mode=mode, border_mode='hard', file_format=file_format)
    print(dataset.tot_frame_num)
    batch, _ = dataset.__getitem__(idx)
    start_idx, end_idx = dataset.context_range(idx)

    return batch, start_idx, end_idx
    
    
    

   


    
    
    
    
    
    
    
    
