import mmcv
from mmcv.image import imread, imwrite
import cv2
from fore_det.inference import inference_detector, init_detector, show_result
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from vad_datasets import unified_dataset_interface
from vad_datasets import bbox_collate, img_tensor2numpy, img_batch_tensor2numpy, frame_size


def imshow_bboxes(img,
                  bboxes,
                  bbox_color=(255, 255, 255),
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


def getObBboxes(img, model, dataset_name, verbose=False):
    if dataset_name == 'UCSDped1':
        score_thr = 0.5
        min_area_thr = 12 * 12
    elif dataset_name == 'UCSDped2':
        score_thr = 0.5
        min_area_thr = 12 * 12
    elif dataset_name == 'avenue':
        score_thr = 0.5
        min_area_thr = 40 * 40
    elif dataset_name == 'ShanghaiTech':
        score_thr = 0.5
        min_area_thr = 8 * 8
    elif dataset_name == 'subway_exit':
        score_thr = 0.5
        min_area_thr = 40 * 40
    elif dataset_name == 'UMN_scene1':
        score_thr = 0.5
        min_area_thr = 10 * 10
    elif dataset_name == 'UMN_scene2':
        score_thr = 0.5
        min_area_thr = 10 * 10
    elif dataset_name == 'UMN_scene3':
        score_thr = 0.5
        min_area_thr = 10 * 10
    else:
        raise NotImplementedError

    result = inference_detector(model, img)

    # bboxes = show_result(img, result, model.CLASSES, score_thr)
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
    bboxes = bboxes[bbox_areas >= min_area_thr, :4]

    if verbose is True:
        imshow_bboxes(img, bboxes)

    return bboxes


def delCoverBboxes(bboxes, dataset_name):
    if dataset_name == 'UCSDped1':
        cover_thr = 0.6
    elif dataset_name == 'UCSDped2':
        cover_thr = 0.6
    elif dataset_name == 'avenue':
        cover_thr = 0.6
    elif dataset_name == 'ShanghaiTech':
        cover_thr = 0.65
    elif dataset_name == 'subway_exit':
        cover_thr = 0.65
    elif dataset_name == 'UMN_scene1' or dataset_name == 'UMN_scene2' or dataset_name == 'UMN_scene3':
        cover_thr = 0.65
    else:
        raise NotImplementedError

    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 4

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    bbox_areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    # Index of bboxes sorted in ascending order by area size
    sort_idx = bbox_areas.argsort()

    keep_idx = []
    for i in range(sort_idx.size):
        # Calculate the point coordinates of the intersection
        x11 = np.maximum(x1[sort_idx[i]], x1[sort_idx[i + 1:]])
        y11 = np.maximum(y1[sort_idx[i]], y1[sort_idx[i + 1:]])
        x22 = np.minimum(x2[sort_idx[i]], x2[sort_idx[i + 1:]])
        y22 = np.minimum(y2[sort_idx[i]], y2[sort_idx[i + 1:]])
        # Calculate the intersection area
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h

        ratios = overlaps / bbox_areas[sort_idx[i]]
        num = ratios[ratios > cover_thr]
        if num.size == 0:
            keep_idx.append(sort_idx[i])

    return bboxes[keep_idx]


def getGdBboxes(cur_img, img_batch, bboxes, dataset_name, verbose=False):
    if dataset_name == 'UCSDped1':
        area_thr = 12 * 12
        binary_thr = 15
        extend = 2
        gauss_mask_size = 3
    elif dataset_name == 'UCSDped2':
        area_thr = 12 * 12
        binary_thr = 18
        extend = 2
        gauss_mask_size = 3
    elif dataset_name == 'avenue':
        area_thr = 40 * 40
        binary_thr = 18
        extend = 2
        gauss_mask_size = 5
    elif dataset_name == 'ShanghaiTech':
        area_thr = 30 * 30
        binary_thr = 15
        extend = 2
        gauss_mask_size = 5
    elif dataset_name == 'subway_exit':
        area_thr = 40 * 40
        binary_thr = 15
        extend = 2
        gauss_mask_size = 5
    elif dataset_name == 'UMN_scene1':
        area_thr = 10 * 10
        binary_thr = 18
        extend = 2
        gauss_mask_size = 5
    elif dataset_name == 'UMN_scene2':
        area_thr = 10 * 10
        binary_thr = 18
        extend = 2
        gauss_mask_size = 5
    elif dataset_name == 'UMN_scene3':
        area_thr = 10 * 10
        binary_thr = 18
        extend = 2
        gauss_mask_size = 5
    else:
        raise NotImplementedError

    sum_grad = 0
    for i in range(img_batch.shape[0] - 1):
        img1 = img_batch[i, :, :, :]
        img2 = img_batch[i + 1, :, :, :]
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
        extend_y1 = np.maximum(0, bbox_int[1] - extend)
        extend_y2 = np.minimum(bbox_int[3] + extend, sum_grad.shape[0])
        extend_x1 = np.maximum(0, bbox_int[0] - extend)
        extend_x2 = np.minimum(bbox_int[2] + extend, sum_grad.shape[1])
        sum_grad[extend_y1:extend_y2 + 1, extend_x1:extend_x2 + 1] = 0
    if verbose is True:
        cv2.imshow('del_ob_bboxes', sum_grad)
        cv2.waitKey(0)

    sum_grad = cv2.cvtColor(sum_grad, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(sum_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        sum_grad = cv2.rectangle(sum_grad, (x, y), (x + w, y + h), 255, 1)
        area = (w + 1) * (h + 1)
        if area > area_thr and w / h < 10 and h / w < 10:
            extend_x1 = np.maximum(0, x - extend)
            extend_y1 = np.maximum(0, y - extend)
            extend_x2 = np.minimum(x + w + extend, sum_grad.shape[1])
            extend_y2 = np.minimum(y + h + extend, sum_grad.shape[0])
            fg_bboxes.append([extend_x1, extend_y1, extend_x2, extend_y2])
            cur_img = cv2.rectangle(cur_img, (extend_x1, extend_y1), (extend_x2, extend_y2), (0, 255, 0), 1)

    if verbose is True:
        cv2.imshow('all_fg_bboxes', sum_grad)
        cv2.waitKey(0)
        cv2.imshow('filter', cur_img)
        cv2.waitKey(0)

    return np.array(fg_bboxes)


def getOfBboxes(cur_of, cur_img, bboxes, dataset_name, verbose=False):
    if dataset_name == 'UCSDped1':
        area_thr = 12 * 12
        binary_thr = 1
        extend = 2
    elif dataset_name == 'UCSDped2':
        area_thr = 12 * 12
        binary_thr = 1
        extend = 2
    elif dataset_name == 'avenue':
        area_thr = 40 * 40
        binary_thr = 1
        extend = 2
    elif dataset_name == 'ShanghaiTech':
        area_thr = 30 * 30
        binary_thr = 1
        extend = 2
    elif dataset_name == 'subway_exit':
        area_thr = 40 * 40
        binary_thr = 1
        extend = 2
    elif dataset_name == 'UMN_scene1':
        area_thr = 10 * 10
        binary_thr = 1
        extend = 2
    elif dataset_name == 'UMN_scene2':
        area_thr = 10 * 10
        binary_thr = 1
        extend = 2
    elif dataset_name == 'UMN_scene3':
        area_thr = 10 * 10
        binary_thr = 1
        extend = 2
    else:
        raise NotImplementedError

    # optical flow intensity map
    cur_of = np.sum(cur_of ** 2, axis=2)
    if verbose is True:
        cv2.imshow('of_intensity', cur_of)
        cv2.waitKey(0)

    # binary map
    cur_of = cv2.threshold(cur_of, binary_thr, 255, cv2.THRESH_BINARY)[1]
    if verbose is True:
        cv2.imshow('binary_map', cur_of)
        cv2.waitKey(0)

    # subtract object detector based RoIs from the map
    for bbox in bboxes:
        bbox_int = bbox.astype(np.int32)
        extend_y1 = np.maximum(0, bbox_int[1] - extend)
        extend_y2 = np.minimum(bbox_int[3] + extend, cur_of.shape[0])
        extend_x1 = np.maximum(0, bbox_int[0] - extend)
        extend_x2 = np.minimum(bbox_int[2] + extend, cur_of.shape[1])
        cur_of[extend_y1:extend_y2 + 1, extend_x1:extend_x2 + 1] = 0
    if verbose is True:
        cv2.imshow('del_ob_bboxes', cur_of)
        cv2.waitKey(0)

    cur_of = cv2.normalize(cur_of, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # contour detection with bounding boxes
    contours, hierarchy = cv2.findContours(cur_of, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cur_of = cv2.rectangle(cur_of, (x, y), (x + w, y + h), 255, 1)
        area = (w + 1) * (h + 1)
        if area > area_thr and w / h < 10 and h / w < 10:
            extend_x1 = np.maximum(0, x - extend)
            extend_y1 = np.maximum(0, y - extend)
            extend_x2 = np.minimum(x + w + extend, cur_of.shape[1])
            extend_y2 = np.minimum(y + h + extend, cur_of.shape[0])
            fg_bboxes.append([extend_x1, extend_y1, extend_x2, extend_y2])
            cur_img = cv2.rectangle(cur_img, (extend_x1, extend_y1), (extend_x2, extend_y2), (0, 255, 0), 1)

    if verbose is True:
        cv2.imshow('all_of_bboxes', cur_of)
        cv2.waitKey(0)
        cv2.imshow('filter_of_bboxes', cur_img)
        cv2.waitKey(0)

    return np.array(fg_bboxes)


if __name__ == '__main__':

    context_frame_num = 1
    idx = 100
    dataset_name = 'avenue'
    mode = 'test'
    dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('../raw_datasets', dataset_name),
                                        context_frame_num=1, mode=mode, border_mode='hard')
    # optical flow dataset
    dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('../optical_flow', dataset_name),
                                         context_frame_num=1, mode=mode, border_mode='hard', file_format='.npy')
    print(dataset.__len__())

    batch, _ = dataset.__getitem__(idx)
    batch2, _ = dataset2.__getitem__(idx)
    print('Extracting bboxes of {}-th frame'.format(idx + 1))
    cur_img = img_tensor2numpy(batch[1])
    cur_of = img_tensor2numpy(batch2[1])

    config_file = '../obj_det_config/cascade_rcnn_r101_fpn_1x.py'
    checkpoint_file = '../obj_det_checkpoints/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    ob_bboxes = getObBboxes(cur_img, model, dataset_name, True)
    ob_bboxes = delCoverBboxes(ob_bboxes, dataset_name)
    print(ob_bboxes.shape)
    # imshow_bboxes(cur_img, ob_bboxes)

    fg_bboxes = getOfBboxes(cur_of, cur_img, ob_bboxes, dataset_name, True)

    if fg_bboxes.shape[0] > 0:
        all_bboxes = np.concatenate((ob_bboxes, fg_bboxes), axis=0)
    else:
        all_bboxes = ob_bboxes
    imshow_bboxes(cur_img, all_bboxes, win_name='all_bboxes')











