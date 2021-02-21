import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from vad_datasets import unified_dataset_interface
from fore_det.inference import init_detector
from vad_datasets import bbox_collate, img_tensor2numpy, img_batch_tensor2numpy, frame_size, cube_to_train_dataset
from fore_det.obj_det_with_motion import imshow_bboxes, get_ap_bboxes, get_mt_bboxes, del_cover_bboxes
from fore_det.simple_patch import get_patch_loc
import cv2
from model.unet import SelfCompleteNet4, SelfCompleteNetFull, SelfCompleteNet1raw1of
import torch.nn as nn
from utils import save_roc_pr_curve_data, calc_block_idx
from configparser import ConfigParser
from helper.visualization_helper import visualize_pair, visualize_batch, visualize_pair_map

#  /*-------------------------------------------------Overall parameter setting-----------------------------------------------------*/
cp = ConfigParser()
cp.read("config.cfg")

dataset_name = cp.get('shared_parameters', 'dataset_name')  # The name of dataset: UCSDped2/avenue/ShanghaiTech.
raw_dataset_dir = cp.get('shared_parameters', 'raw_dataset_dir')  # Fixed
foreground_extraction_mode = cp.get('shared_parameters', 'foreground_extraction_mode')  # Foreground extraction method: obj_det_with_motion/obj_det/simple_patch/frame.
data_root_dir = cp.get('shared_parameters', 'data_root_dir')  # Fixed: A folder that stores the data such as foreground produced by the program.
modality = cp.get('shared_parameters', 'modality')  # Fixed
mode = cp.get('test_parameters', 'mode')  # Fixed
method = cp.get('shared_parameters', 'method')  # Fixed
try:
    patch_size = cp.getint(dataset_name, 'patch_size')  # Resize the foreground bounding boxes.
    test_block_mode = cp.getint(dataset_name, 'test_block_mode')  # Fixed
    motionThr = cp.getfloat(dataset_name, 'motionThr')  # Fixed
    # Define h_block * w_block sub-regions of video frames for localized testing
    h_block = cp.getint(dataset_name, 'h_block')  # Localized
    w_block = cp.getint(dataset_name, 'w_block')  # Localized

    # Set 'bbox_save=False' and 'foreground_saved=False' at first to calculate and store the bboxes and foreground,
    # then set them to True to load the stored bboxes and foreground directly, if the foreground parameters remain unchanged.
    bbox_saved = cp.getboolean(dataset_name, 'test_bbox_saved')
    foreground_saved = cp.getboolean(dataset_name, 'test_foreground_saved')
except:
    raise NotImplementedError

#  /*--------------------------------------------------Foreground localization-----------------------------------------------------*/
config_file = 'fore_det/obj_det_config/cascade_rcnn_r101_fpn_1x.py'
checkpoint_file = 'fore_det/obj_det_checkpoints/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth'

# Set dataset for foreground localization.
dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(raw_dataset_dir, dataset_name),
                                    context_frame_num=1, mode=mode, border_mode='hard')

if not bbox_saved:
    # Build the object detector from a config file and a checkpoint file.
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    all_bboxes = list()
    for idx in range(dataset.__len__()):
        batch, _ = dataset.__getitem__(idx)
        print('Extracting bboxes of {}-th frame'.format(idx + 1))
        cur_img = img_tensor2numpy(batch[1])

        if foreground_extraction_mode == 'obj_det_with_motion':
            # A coarse detection of bboxes by pretrained object detector.
            ap_bboxes = get_ap_bboxes(cur_img, model, dataset_name, verbose=False)

            # Delete overlapping appearance based bounding boxes.
            ap_bboxes = del_cover_bboxes(ap_bboxes, dataset_name)
            # imshow_bboxes(cur_img, ap_bboxes, win_name='kept ap based bboxes')

            # Further foreground detection by motion.
            mt_bboxes = get_mt_bboxes(cur_img, img_batch_tensor2numpy(batch), ap_bboxes, dataset_name, verbose=False)
            if mt_bboxes.shape[0] > 0:
                cur_bboxes = np.concatenate((ap_bboxes, mt_bboxes), axis=0)
            else:
                cur_bboxes = ap_bboxes
        elif foreground_extraction_mode == 'obj_det':
            # A coarse detection of bboxes by pretrained object detector.
            ap_bboxes = get_ap_bboxes(cur_img, model, dataset_name)
            cur_bboxes = del_cover_bboxes(ap_bboxes, dataset_name)
        elif foreground_extraction_mode == 'simple_patch':
            patch_num_list = [(3, 4), (6, 8)]
            cur_bboxes = list()
            for h_num, w_num in patch_num_list:
                cur_bboxes.append(get_patch_loc(frame_size[dataset_name][0], frame_size[dataset_name][1], h_num, w_num))
            cur_bboxes = np.concatenate(cur_bboxes, axis=0)
        elif foreground_extraction_mode == 'frame':
            cur_bboxes = list()
            cur_bboxes.append([0, 0, cur_img.shape[1], cur_img.shape[0]])
            cur_bboxes = np.array(cur_bboxes)
        else:
            raise NotImplementedError

        # imshow_bboxes(cur_img, cur_bboxes, win_name='all foreground bboxes')
        all_bboxes.append(cur_bboxes)
    np.save(os.path.join(dataset.dir, 'bboxes_test_{}.npy'.format(foreground_extraction_mode)), all_bboxes)
    print('bboxes for testing data saved!')
else:
    all_bboxes = np.load(os.path.join(dataset.dir, 'bboxes_test_{}.npy'.format(foreground_extraction_mode)), allow_pickle=True)
    print('bboxes for testing data loaded!')

#  /*--------------------------------------------------Foreground extraction--------------------------------------------------------*/
if not foreground_saved:
    context_frame_num = cp.getint(method, 'context_frame_num')
    context_of_num = cp.getint(method, 'context_of_num')
    border_mode = cp.get(method, 'border_mode')
    if modality == 'raw_datasets':
        file_format = frame_size[dataset_name][2]
    elif modality == 'raw2flow':
        file_format1 = frame_size[dataset_name][2]
        file_format2 = '.npy'
    else:
        file_format = '.npy'

    if modality == 'raw2flow':
        dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                            context_frame_num=context_frame_num, mode=mode, border_mode=border_mode,
                                            all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format1)
        dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name),
                                             context_frame_num=context_of_num, mode=mode, border_mode=border_mode,
                                             all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format2)
    else:
        dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                            context_frame_num=context_frame_num, mode=mode, border_mode=border_mode,
                                            all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format1)

    if dataset_name == 'ShanghaiTech':
        np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'scene_idx.npy'), dataset.scene_idx)
        scene_idx = dataset.scene_idx  # 1 scene

    foreground_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]
    if modality == 'raw2flow':
        foreground_set2 = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]
    foreground_bbox_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]

    h_step, w_step = frame_size[dataset_name][0] / h_block, frame_size[dataset_name][1] / w_block

    for idx in range(dataset.__len__()):
        batch, _ = dataset.__getitem__(idx)
        if modality == 'raw2flow':
            batch2, _ = dataset2.__getitem__(idx)
        print('Extracting foreground in {}-th batch, {} in total'.format(idx + 1, dataset.__len__() // 1))
        cur_bboxes = all_bboxes[idx]
        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            if modality == 'raw2flow':
                batch2 = img_batch_tensor2numpy(batch2)

            if modality == 'optical_flow':
                if len(batch.shape) == 4:
                    mag = np.sum(np.sum(np.sum(batch ** 2, axis=3), axis=2), axis=1)
                else:
                    mag = np.mean(np.sum(np.sum(np.sum(batch ** 2, axis=4), axis=3), axis=2), axis=1)
            elif modality == 'raw2flow':
                if len(batch2.shape) == 4:
                    mag = np.sum(np.sum(np.sum(batch2 ** 2, axis=3), axis=2), axis=1)
                else:
                    mag = np.mean(np.sum(np.sum(np.sum(batch2 ** 2, axis=4), axis=3), axis=2), axis=1)
            else:
                mag = np.ones(batch.shape[0]) * 10000

            for idx_bbox in range(cur_bboxes.shape[0]):
                if mag[idx_bbox] > motionThr:
                    all_blocks = calc_block_idx(cur_bboxes[idx_bbox, 0], cur_bboxes[idx_bbox, 2], cur_bboxes[idx_bbox, 1], cur_bboxes[idx_bbox, 3], h_step, w_step, mode=test_block_mode)
                    for (h_block_idx, w_block_idx) in all_blocks:
                        foreground_set[idx][h_block_idx][w_block_idx].append(batch[idx_bbox])
                        if modality == 'raw2flow':
                            foreground_set2[idx][h_block_idx][w_block_idx].append(batch2[idx_bbox])
                        foreground_bbox_set[idx][h_block_idx][w_block_idx].append(cur_bboxes[idx_bbox])

    foreground_set = [[[np.array(foreground_set[ii][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]
    if modality == 'raw2flow':
        foreground_set2 = [[[np.array(foreground_set2[ii][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]

    foreground_bbox_set = [[[np.array(foreground_bbox_set[ii][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]
    if modality == 'raw2flow':
        np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_test_{}-raw.npy'.format(foreground_extraction_mode)), foreground_set)
        np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_test_{}-flow.npy'.format(foreground_extraction_mode)), foreground_set2)
    else:
        np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_test_{}.npy'.format(foreground_extraction_mode)), foreground_set)
    np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_bbox_test_{}.npy'.format(foreground_extraction_mode)), foreground_bbox_set)
    print('foreground for testing data saved!')
else:
    if dataset_name == 'ShanghaiTech':
        scene_idx = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'scene_idx.npy'))
    if modality == 'raw2flow':
        foreground_set = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_test_{}-raw.npy'.format(foreground_extraction_mode)), allow_pickle=True)
        foreground_set2 = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_test_{}-flow.npy'.format(foreground_extraction_mode)), allow_pickle=True)
    else:
        foreground_set = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_test_{}.npy'.format(foreground_extraction_mode)), allow_pickle=True)

    foreground_bbox_set = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_bbox_test_{}.npy'.format(foreground_extraction_mode)), allow_pickle=True)
    print('foreground for testing data loaded!')

#  /*-------------------------------------------------Abnormal event detection-----------------------------------------------------*/
results_dir = 'results'
scores_saved = cp.getboolean(dataset_name, 'scores_saved')
big_number = 100000
if scores_saved is False:
    if method == 'SelfComplete':
        h, w, _, sn = frame_size[dataset_name]
        border_mode = cp.get(method, 'border_mode')
        if border_mode == 'predict':
            tot_frame_num = cp.getint(method, 'context_frame_num') + 1
            tot_of_num = cp.getint(method, 'context_of_num') + 1
        else:
            tot_frame_num = 2 * cp.getint(method, 'context_frame_num') + 1
            tot_of_num = 2 * cp.getint(method, 'context_of_num') + 1
        rawRange = cp.getint(method, 'rawRange')
        if rawRange >= tot_frame_num:  # If rawRange is out of the range, use all frames.
            rawRange = None
        useFlow = cp.getboolean(method, 'useFlow')
        padding = cp.getboolean(method, 'padding')

        assert modality == 'raw2flow'
        score_func = nn.MSELoss(reduce=False)

        if tot_of_num == 1:
            network_architecture = SelfCompleteNet4(features_root=cp.getint(method, 'nf'), tot_raw_num=tot_frame_num, tot_of_num=tot_of_num,
                                                    border_mode=border_mode, rawRange=rawRange, useFlow=useFlow, padding=padding)
        elif tot_of_num == 5:
            network_architecture = SelfCompleteNetFull(features_root=cp.getint(method, 'nf'), tot_raw_num=tot_frame_num, tot_of_num=tot_of_num,
                                                       border_mode=border_mode, rawRange=rawRange, useFlow=useFlow, padding=padding)
        else:
            NotImplementedError
        assert tot_frame_num == 5

        pixel_result_dir = os.path.join(results_dir, dataset_name, 'score_mask')
        os.makedirs(pixel_result_dir, exist_ok=True)  # A folder to store frame pixel results.

        # Load saved models.
        model_weights = torch.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'model_{}_{}.npy'.format(foreground_extraction_mode, method)))
        if dataset_name == 'ShanghaiTech':
            model_set = [[[[] for ww in range(len(model_weights[ss][hh]))] for hh in range(len(model_weights[ss]))] for ss in range(len(model_weights))]
            for ss in range(len(model_weights)):
                for hh in range(len(model_weights[ss])):
                    for ww in range(len(model_weights[ss][hh])):
                        if len(model_weights[ss][hh][ww]) > 0:
                            cur_model = torch.nn.DataParallel(network_architecture, device_ids=[0]).cuda()
                            cur_model.load_state_dict(model_weights[ss][hh][ww][0])
                            model_set[ss][hh][ww].append(cur_model.eval())

            # Get training score statistics.
            raw_training_scores_set = torch.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'raw_training_scores_{}_{}.npy'.format(foreground_extraction_mode, method)))
            of_training_scores_set = torch.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'of_training_scores_{}_{}.npy'.format(foreground_extraction_mode, method)))

            # Calculate mean and std of training scores.
            raw_stats_set = [[[(np.mean(raw_training_scores_set[ss][hh][ww]), np.std(raw_training_scores_set[ss][hh][ww])) for ww in range(w_block)] for hh in range(h_block)] for ss in range(len(model_weights))]
            if useFlow:
                of_stats_set = [[[(np.mean(of_training_scores_set[ss][hh][ww]), np.std(of_training_scores_set[ss][hh][ww])) for ww in range(w_block)] for hh in range(h_block)] for ss in range(len(model_weights))]
            del raw_training_scores_set, of_training_scores_set
        else:
            model_set = [[[] for ww in range(len(model_weights[hh]))] for hh in range(len(model_weights))]
            for hh in range(len(model_weights)):
                for ww in range(len(model_weights[hh])):
                    if len(model_weights[hh][ww]) > 0:
                        cur_model = torch.nn.DataParallel(network_architecture, device_ids=[0]).cuda()
                        cur_model.load_state_dict(model_weights[hh][ww][0])
                        model_set[hh][ww].append(cur_model.eval())

            # Get training score statistics.
            raw_training_scores_set = torch.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'raw_training_scores_{}_{}.npy'.format(foreground_extraction_mode, method)))
            of_training_scores_set = torch.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'of_training_scores_{}_{}.npy'.format(foreground_extraction_mode, method)))
            
            # Calculate mean and std of training scores.
            raw_stats_set = [[(np.mean(raw_training_scores_set[hh][ww]), np.std(raw_training_scores_set[hh][ww])) for ww in range(len(model_weights[hh]))] for hh in range(len(model_weights))]
            if useFlow:
                of_stats_set = [[(np.mean(of_training_scores_set[hh][ww]), np.std(of_training_scores_set[hh][ww])) for ww in range(len(model_weights[hh]))] for hh in range(len(model_weights))]
            del raw_training_scores_set, of_training_scores_set

        # Calculate anomaly scores for each video event (frame).
        for frame_idx in range(len(foreground_set)):
            print('Calculating scores for {}-th frame'.format(frame_idx))
            cur_data_set = foreground_set[frame_idx]
            cur_data_set2 = foreground_set2[frame_idx]
            cur_bboxes = foreground_bbox_set[frame_idx]
            # Normal: no objects in this block.
            cur_pixel_results = -1 * np.ones(shape=(h, w)) * big_number
            for h_idx in range(len(cur_data_set)):
                for w_idx in range(len(cur_data_set[h_idx])):
                    if len(cur_data_set[h_idx][w_idx]) > 0:

                        if dataset_name == 'ShanghaiTech':
                            if len(model_set[scene_idx[frame_idx] - 1][h_idx][w_idx]) > 0:
                                cur_model = model_set[scene_idx[frame_idx] - 1][h_idx][w_idx][0]
                                cur_dataset = cube_to_train_dataset(cur_data_set[h_idx][w_idx], target=cur_data_set2[h_idx][w_idx])
                                cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=cur_data_set[h_idx][w_idx].shape[0], shuffle=False)
                                for idx, (inputs, of_targets_all, _) in enumerate(cur_dataloader):
                                    inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                                    of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)
                                    
                                    of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)

                                    if useFlow:
                                        of_scores = score_func(of_targets, of_outputs).cpu().data.numpy()
                                        of_scores = np.sum(np.sum(np.sum(of_scores, axis=3), axis=2), axis=1)  # MSE score.

                                    raw_scores = score_func(raw_targets, raw_outputs).cpu().data.numpy()
                                    raw_scores = np.sum(np.sum(np.sum(raw_scores, axis=3), axis=2), axis=1)  # MSE score.

                                    # Normalize scores using training scores.
                                    raw_scores = (raw_scores - raw_stats_set[scene_idx[frame_idx] - 1][h_idx][w_idx][0]) / raw_stats_set[scene_idx[frame_idx] - 1][h_idx][w_idx][1]
                                    if useFlow:
                                        of_scores = (of_scores - of_stats_set[scene_idx[frame_idx] - 1][h_idx][w_idx][0]) / of_stats_set[scene_idx[frame_idx] - 1][h_idx][w_idx][1]

                                    if useFlow:
                                        scores = cp.getfloat(method, 'w_raw') * raw_scores + cp.getfloat(method, 'w_of') * of_scores
                                    else:
                                        scores = cp.getfloat(method, 'w_raw') * raw_scores
                            else:
                                # Anomaly: No object in training set while objects occur in this block.
                                scores = np.ones(cur_data_set[h_idx][w_idx].shape[0], ) * big_number
                        else:
                            if len(model_set[h_idx][w_idx]) > 0:
                                cur_model = model_set[h_idx][w_idx][0]
                                cur_dataset = cube_to_train_dataset(cur_data_set[h_idx][w_idx], target=cur_data_set2[h_idx][w_idx])
                                cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=cur_data_set[h_idx][w_idx].shape[0], shuffle=False)
                                for idx, (inputs, of_targets_all, _) in enumerate(cur_dataloader):
                                    inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                                    of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)
                                    of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)
                                    
                                    # Visualization.
                                    # max_num = 30
                                    # visualize_pair_map(
                                    #     batch_1=img_batch_tensor2numpy(raw_targets.cpu().detach()[:max_num, 6:9, :, :]),
                                    #     batch_2=img_batch_tensor2numpy(raw_outputs.cpu().detach()[:max_num, 6:9, :, :]))
                                    # visualize_pair(
                                    #     batch_1=img_batch_tensor2numpy(of_targets.cpu().detach()[:max_num, 4:6, :, :]),
                                    #     batch_2=img_batch_tensor2numpy(of_outputs.cpu().detach()[:max_num, 4:6, :, :]))

                                    if useFlow:
                                        of_scores = score_func(of_targets, of_outputs).cpu().data.numpy()
                                        of_scores = np.sum(np.sum(np.sum(of_scores, axis=3), axis=2), axis=1)  # MSE score.

                                    raw_scores = score_func(raw_targets, raw_outputs).cpu().data.numpy()
                                    raw_scores = np.sum(np.sum(np.sum(raw_scores, axis=3), axis=2), axis=1)  # MSE score.
                                        
                                    # Normalize scores using training scores.
                                    raw_scores = (raw_scores - raw_stats_set[h_idx][w_idx][0]) / raw_stats_set[h_idx][w_idx][1]
                                    if useFlow:
                                        of_scores = (of_scores - of_stats_set[h_idx][w_idx][0]) / of_stats_set[h_idx][w_idx][1]
                                        
                                    if useFlow:
                                        scores = cp.getfloat(method, 'w_raw') * raw_scores + cp.getfloat(method, 'w_of') * of_scores
                                    else:
                                        scores = cp.getfloat(method, 'w_raw') * raw_scores   
                            else:
                                # Anomaly: No object in training set while objects occur in this block.
                                scores = np.ones(cur_data_set[h_idx][w_idx].shape[0], ) * big_number

                        for m in range(scores.shape[0]):
                            cur_score_mask = -1 * np.ones(shape=(h, w)) * big_number
                            cur_score = scores[m]
                            bbox = cur_bboxes[h_idx][w_idx][m]
                            x_min, x_max = np.int(np.ceil(bbox[0])), np.int(np.ceil(bbox[2]))
                            y_min, y_max = np.int(np.ceil(bbox[1])), np.int(np.ceil(bbox[3]))
                            cur_score_mask[y_min:y_max, x_min:x_max] = cur_score
                            cur_pixel_results = np.max(np.concatenate([cur_pixel_results[:, :, np.newaxis], cur_score_mask[:, :, np.newaxis]], axis=2), axis=2)
            torch.save(cur_pixel_results, os.path.join(pixel_result_dir, '{}'.format(frame_idx))) 
    else:
        raise NotImplementedError

#  /*-------------------------------------------------------Evaluation-----------------------------------------------------------*/
criterion = 'frame'
batch_size = 1
# Set dataset for evaluation.
dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(raw_dataset_dir, dataset_name), context_frame_num=0, mode=mode, border_mode='hard')
dataset_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=bbox_collate(mode).collate)

print('Evaluating {} by {}-criterion:'.format(dataset_name, criterion))
if criterion == 'frame':
    if dataset_name == 'ShanghaiTech':
        all_frame_scores = [[] for si in set(dataset.scene_idx)]
        all_targets = [[] for si in set(dataset.scene_idx)]
        for idx, (_, target) in enumerate(dataset_loader):
            print('Processing {}-th frame'.format(idx))
            cur_pixel_results = torch.load(os.path.join(results_dir, dataset_name, 'score_mask', '{}'.format(idx)))
            all_frame_scores[scene_idx[idx] - 1].append(cur_pixel_results.max())
            all_targets[scene_idx[idx] - 1].append(target[0].numpy().max())
        all_frame_scores = [np.array(all_frame_scores[si]) for si in range(dataset.scene_num)]
        all_targets = [np.array(all_targets[si]) for si in range(dataset.scene_num)]
        all_targets = [all_targets[si] > 0 for si in range(dataset.scene_num)]
        results = [save_roc_pr_curve_data(all_frame_scores[si], all_targets[si], os.path.join(results_dir, dataset_name,
                   '{}_{}_{}_frame_results_scene_{}.npz'.format(modality, foreground_extraction_mode, method, si + 1))) for si in range(dataset.scene_num)]
        results = np.array(results).mean()
        print('Average frame-level AUC is {}'.format(results))
    else:
        all_frame_scores = list()
        all_targets = list()
        for idx, (_, target) in enumerate(dataset_loader):
            print('Processing {}-th frame'.format(idx))
            cur_pixel_results = torch.load(os.path.join(results_dir, dataset_name, 'score_mask', '{}'.format(idx)))
            all_frame_scores.append(cur_pixel_results.max())
            all_targets.append(target[0].numpy().max())
        all_frame_scores = np.array(all_frame_scores)                   
        all_targets = np.array(all_targets)
        all_targets = all_targets > 0
        results_path = os.path.join(results_dir, dataset_name, '{}_{}_{}_frame_results.npz'.format(modality, foreground_extraction_mode, method))
        print('Results written to {}:'.format(results_path))
        auc = save_roc_pr_curve_data(all_frame_scores, all_targets, results_path)
elif criterion == 'pixel':
    if dataset_name != 'ShanghaiTech':
        all_pixel_scores = list()
        all_targets = list()
        thr = 0.4
        for idx, (_, target) in enumerate(dataset_loader):
            print('Processing {}-th frame'.format(idx))
            cur_pixel_results = torch.load(os.path.join(results_dir, dataset_name, 'score_mask', '{}'.format(idx)))
            target_mask = target[0].numpy()
            all_targets.append(target[0].numpy().max())
            if all_targets[-1] > 0:
                cur_effective_scores = cur_pixel_results[target_mask > 0]
                sorted_score = np.sort(cur_effective_scores)
                cut_off_idx = np.int(np.round((1 - thr) * cur_effective_scores.shape[0]))
                cut_off_score = cur_effective_scores[cut_off_idx]
            else:
                cut_off_score = cur_pixel_results.max()
            all_pixel_scores.append(cut_off_score)
        all_frame_scores = np.array(all_pixel_scores)
        all_targets = np.array(all_targets)
        all_targets = all_targets > 0
        results_path = os.path.join(results_dir, dataset_name,
                                    '{}_{}_{}_pixel_results.npz'.format(modality, foreground_extraction_mode, method))
        print('Results written to {}:'.format(results_path))
        results = save_roc_pr_curve_data(all_frame_scores, all_targets, results_path)
    else:
        raise NotImplementedError
else:
    raise NotImplementedError
