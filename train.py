import numpy as np
import os
from torch.utils.data import DataLoader
from vad_datasets import unified_dataset_interface, cube_to_train_dataset
from fore_det.inference import init_detector
from vad_datasets import bbox_collate, img_tensor2numpy, img_batch_tensor2numpy, frame_size
from fore_det.obj_det_with_motion import imshow_bboxes, get_ap_bboxes, get_mt_bboxes, del_cover_bboxes
from fore_det.simple_patch import get_patch_loc
import cv2
from helper.misc import AverageMeter
import torch
from model.unet import SelfCompleteNet4, SelfCompleteNetFull, SelfCompleteNet1raw1of
import torch.optim as optim
import torch.nn as nn
from configparser import ConfigParser
from utils import calc_block_idx

#  /*-------------------------------------------------Overall parameter setting-----------------------------------------------------*/
cp = ConfigParser()
cp.read("config.cfg")

dataset_name = cp.get('shared_parameters', 'dataset_name')  # The name of dataset: UCSDped2/avenue/ShanghaiTech.
raw_dataset_dir = cp.get('shared_parameters', 'raw_dataset_dir')  # Fixed
foreground_extraction_mode = cp.get('shared_parameters', 'foreground_extraction_mode')  # Foreground extraction method: obj_det_with_motion/obj_det/simple_patch/frame.
data_root_dir = cp.get('shared_parameters', 'data_root_dir')  # Fixed: A folder that stores the data such as foreground produced by the program.
modality = cp.get('shared_parameters', 'modality')  # Fixed
mode = cp.get('train_parameters', 'mode')  # Fixed
method = cp.get('shared_parameters', 'method')  # Fixed
try:
    patch_size = cp.getint(dataset_name, 'patch_size')  # Resize the foreground bounding boxes.
    train_block_mode = cp.getint(dataset_name, 'train_block_mode')  # Fixed
    motionThr = cp.getfloat(dataset_name, 'motionThr')  # Fixed
    # Define h_block * w_block sub-regions of video frames for localized training
    h_block = cp.getint(dataset_name, 'h_block')  # Localized
    w_block = cp.getint(dataset_name, 'w_block')  # Localized

    # Set 'bbox_save=False' and 'foreground_saved=False' at first to calculate and store the bboxes and foreground,
    # then set them to True to load the stored bboxes and foreground directly, if the foreground parameters remain unchanged.
    bbox_saved = cp.getboolean(dataset_name, 'train_bbox_saved')
    foreground_saved = cp.getboolean(dataset_name, 'train_foreground_saved')
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
            # A coarse detection of bboxes by pretrained object detector
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
    np.save(os.path.join(dataset.dir, 'bboxes_train_{}.npy'.format(foreground_extraction_mode)), all_bboxes)
    print('bboxes for training data saved!')
else:
    all_bboxes = np.load(os.path.join(dataset.dir, 'bboxes_train_{}.npy'.format(foreground_extraction_mode)), allow_pickle=True)
    print('bboxes for training data loaded!')

# /*--------------------------------------------------Foreground extraction--------------------------------------------------------*/
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
        dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(modality, dataset_name),
                                            context_frame_num=context_frame_num, mode=mode, border_mode=border_mode,
                                            all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format)

    if dataset_name == 'ShanghaiTech':
        foreground_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in range(dataset.scene_num)]
        if modality == 'raw2flow':
            foreground_set2 = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in range(dataset.scene_num)]
    else:
        foreground_set = [[[] for ww in range(w_block)] for hh in range(h_block)]
        if modality == 'raw2flow':
            foreground_set2 = [[[] for ww in range(w_block)] for hh in range(h_block)]

    h_step, w_step = frame_size[dataset_name][0] / h_block, frame_size[dataset_name][1] / w_block

    # ShanghaiTech training set is too large and needs to be stored in segments (Memory=64G).
    if dataset_name == 'ShanghaiTech' and modality == 'raw2flow':
        randIdx = np.random.permutation(dataset.__len__())
        cout = 0
        segIdx = 0
        saveSegNum = cp.getint(dataset_name, 'saveSegNum')

    for iidx in range(dataset.__len__()):
        if dataset_name == 'ShanghaiTech' and modality == 'raw2flow':
            idx = randIdx[iidx]
            cout += 1
        else:
            idx = iidx

        batch, _ = dataset.__getitem__(idx)
        if modality == 'raw2flow':
            batch2, _ = dataset2.__getitem__(idx)

        if dataset_name == 'ShanghaiTech':
            print('Extracting foreground in {}-th batch, {} in total, scene: {}'.format(iidx + 1, dataset.__len__() // 1, dataset.scene_idx[idx]))
        else:
            print('Extracting foreground in {}-th batch, {} in total'.format(iidx + 1, dataset.__len__() // 1))

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
                    all_blocks = calc_block_idx(cur_bboxes[idx_bbox, 0], cur_bboxes[idx_bbox, 2], cur_bboxes[idx_bbox, 1], cur_bboxes[idx_bbox, 3], h_step, w_step, mode=train_block_mode)
                    for (h_block_idx, w_block_idx) in all_blocks:
                        if dataset_name == 'ShanghaiTech':
                            foreground_set[dataset.scene_idx[idx] - 1][h_block_idx][w_block_idx].append(batch[idx_bbox])
                            if modality == 'raw2flow':
                                foreground_set2[dataset.scene_idx[idx] - 1][h_block_idx][w_block_idx].append(batch2[idx_bbox])
                        else:
                            foreground_set[h_block_idx][w_block_idx].append(batch[idx_bbox])
                            if modality == 'raw2flow':
                                foreground_set2[h_block_idx][w_block_idx].append(batch2[idx_bbox])

        if dataset_name == 'ShanghaiTech' and modality == 'raw2flow':
            if cout == saveSegNum:
                # ShanghaiTech training set is too large and needs to be stored in segments
                foreground_set = [[[np.array(foreground_set[ss][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ss in range(dataset.scene_num)]
                foreground_set2 = [[[np.array(foreground_set2[ss][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ss in range(dataset.scene_num)]
                np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}_seg_{}-raw.npy'.format(foreground_extraction_mode, segIdx)), foreground_set)
                np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}_seg_{}-flow.npy'.format(foreground_extraction_mode, segIdx)), foreground_set2)
                del foreground_set, foreground_set2

                cout = 0
                segIdx += 1
                foreground_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in range(dataset.scene_num)]
                foreground_set2 = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in range(dataset.scene_num)]

    if dataset_name == 'ShanghaiTech':
        if modality != 'raw2flow':
            foreground_set = [[[np.array(foreground_set[ss][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ss in range(dataset.scene_num)]
            np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}.npy'.format(foreground_extraction_mode)), foreground_set)
        else:
            if dataset.__len__() % saveSegNum != 0:
                foreground_set = [[[np.array(foreground_set[ss][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ss in range(dataset.scene_num)]
                foreground_set2 = [[[np.array(foreground_set2[ss][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ss in range(dataset.scene_num)]
                np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}_seg_{}-raw.npy'.format(foreground_extraction_mode, segIdx)), foreground_set)
                np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}_seg_{}-flow.npy'.format(foreground_extraction_mode, segIdx)), foreground_set2)
    else:
        if modality == 'raw2flow':
            foreground_set = [[np.array(foreground_set[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
            np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}-raw.npy'.format(foreground_extraction_mode)), foreground_set)
            foreground_set2 = [[np.array(foreground_set2[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
            np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}-flow.npy'.format(foreground_extraction_mode)), foreground_set2)
        else:
            foreground_set = [[np.array(foreground_set[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
            np.save(os.path.join(data_root_dir, modality, dataset_name+'_'+'foreground_train_{}.npy'.format(foreground_extraction_mode)), foreground_set)
    print('foreground for training data saved!')
else:
    if dataset_name != 'ShanghaiTech':
        if modality == 'raw2flow':
            foreground_set = np.load(os.path.join(data_root_dir, modality, dataset_name+'_'+'foreground_train_{}-raw.npy'.format(foreground_extraction_mode)), allow_pickle=True)
            foreground_set2 = np.load(os.path.join(data_root_dir, modality, dataset_name+'_'+'foreground_train_{}-flow.npy'.format(foreground_extraction_mode)), allow_pickle=True)
        else:
            foreground_set = np.load(os.path.join(data_root_dir, modality, dataset_name+'_'+'foreground_train_{}.npy'.format(foreground_extraction_mode)), allow_pickle=True)
        print('foreground for training data loaded!')
    else:
        if modality != 'raw2flow':
            foreground_set = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}.npy'.format(foreground_extraction_mode)), allow_pickle=True)

#  /*-----------------------------------------------Normal video event modeling---------------------------------------------------*/
if method == 'SelfComplete':
    loss_func = nn.MSELoss()
    epochs = cp.getint(method, 'epochs')
    batch_size = cp.getint(method, 'batch_size')
    useFlow = cp.getboolean(method, 'useFlow')
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
    padding = cp.getboolean(method, 'padding')
    lambda_raw = cp.getfloat(method, 'lambda_raw')
    lambda_of = cp.getfloat(method, 'lambda_of')
    assert modality == 'raw2flow'

    if tot_of_num == 1:
        network_architecture = SelfCompleteNet4(features_root=cp.getint(method, 'nf'), tot_raw_num=tot_frame_num, tot_of_num=tot_of_num,
                                                border_mode=border_mode, rawRange=rawRange, useFlow=useFlow, padding=padding)
    elif tot_of_num == 5:
        network_architecture = SelfCompleteNetFull(features_root=cp.getint(method, 'nf'), tot_raw_num=tot_frame_num, tot_of_num=tot_of_num,
                                                   border_mode=border_mode, rawRange=rawRange, useFlow=useFlow, padding=padding)
    else:
        NotImplementedError
    assert tot_frame_num == 5

    if dataset_name == 'ShanghaiTech':
        model_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in range(frame_size[dataset_name][-1])]
        raw_training_scores_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in range(frame_size[dataset_name][-1])]
        of_training_scores_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in range(frame_size[dataset_name][-1])]
    else:
        model_set = [[[] for ww in range(len(foreground_set[hh]))] for hh in range(len(foreground_set))]
        raw_training_scores_set = [[[] for ww in range(len(foreground_set[hh]))] for hh in range(len(foreground_set))]
        of_training_scores_set = [[[] for ww in range(len(foreground_set[hh]))] for hh in range(len(foreground_set))]

    # Prepare training data in current block
    if dataset_name == 'ShanghaiTech':
        saveSegNum = cp.getint(dataset_name, 'saveSegNum')
        totSegNum = np.int(np.ceil(dataset.__len__() / saveSegNum))
        for s_idx in range(len(model_set)):
            for h_idx in range(len(model_set[s_idx])):
                for w_idx in range(len(model_set[s_idx][h_idx])):
                    raw_losses = AverageMeter()
                    of_losses = AverageMeter()
                    # Prepare UNet model and training parameters for current block
                    cur_model = torch.nn.DataParallel(network_architecture).cuda()
                    optimizer = optim.Adam(cur_model.parameters(), eps=1e-7, weight_decay=0.000)
                    cur_model.train()
                    for epoch in range(epochs):
                        for segIdx in range(totSegNum):
                            foreground_set = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}_seg_{}-raw.npy'.format(foreground_extraction_mode, segIdx)), allow_pickle=True)
                            foreground_set2 = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}_seg_{}-flow.npy'.format(foreground_extraction_mode, segIdx)), allow_pickle=True)
                            cur_training_data = foreground_set[s_idx][h_idx][w_idx]
                            cur_training_data2 = foreground_set2[s_idx][h_idx][w_idx]
                            cur_dataset = cube_to_train_dataset(cur_training_data, target=cur_training_data2)
                            cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=True)

                            for idx, (inputs, of_targets_all, _) in enumerate(cur_dataloader):
                                inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                                of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)

                                of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)

                                loss_raw = loss_func(raw_targets.detach(), raw_outputs)
                                if useFlow:
                                    loss_of = loss_func(of_targets.detach(), of_outputs)

                                if useFlow:
                                    loss = lambda_raw * loss_raw + lambda_of * loss_of
                                else:
                                    loss = loss_raw

                                raw_losses.update(loss_raw.item(), inputs.size(0))
                                if useFlow:
                                    of_losses.update(loss_of.item(), inputs.size(0))
                                else:
                                    of_losses.update(0., inputs.size(0))

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                                if idx % 5 == 0:
                                    print('Block: ({}, {}), epoch {}, seg {}, batch {} of {}, raw loss: {}, of loss: {}'.format(
                                        h_idx, w_idx, epoch, segIdx, idx, cur_dataset.__len__() // batch_size, raw_losses.avg,
                                        of_losses.avg))

                    model_set[s_idx][h_idx][w_idx].append(cur_model.state_dict())

                    # A forward pass to store the training scores of optical flow and raw datasets respectively.
                    for segIdx in range(totSegNum):
                        foreground_set = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}_seg_{}-raw.npy'.format(foreground_extraction_mode, segIdx)))
                        foreground_set2 = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}_seg_{}-flow.npy'.format(foreground_extraction_mode, segIdx)))
                        cur_training_data = foreground_set[s_idx][h_idx][w_idx]
                        cur_training_data2 = foreground_set2[s_idx][h_idx][w_idx]
                        cur_dataset = cube_to_train_dataset(cur_training_data, target=cur_training_data2)

                        forward_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=False)
                        score_func = nn.MSELoss(reduce=False)
                        cur_model.eval()
                        for idx, (inputs, of_targets_all, _) in enumerate(forward_dataloader):
                            inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                            of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)

                            of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)
                            raw_scores = score_func(raw_targets, raw_outputs).cpu().data.numpy()
                            raw_scores = np.sum(np.sum(np.sum(raw_scores, axis=3), axis=2), axis=1)  # mse
                            raw_training_scores_set[s_idx][h_idx][w_idx].append(raw_scores)
                            if useFlow:
                                of_scores = score_func(of_targets, of_outputs).cpu().data.numpy()
                                of_scores = np.sum(np.sum(np.sum(of_scores, axis=3), axis=2), axis=1)  # mse
                                of_training_scores_set[s_idx][h_idx][w_idx].append(of_scores)

                    raw_training_scores_set[s_idx][h_idx][w_idx] = np.concatenate(raw_training_scores_set[s_idx][h_idx][w_idx], axis=0)
                    if useFlow:
                        of_training_scores_set[s_idx][h_idx][w_idx] = np.concatenate(of_training_scores_set[s_idx][h_idx][w_idx], axis=0)
                    del cur_model, raw_losses, of_losses

        torch.save(raw_training_scores_set, os.path.join(data_root_dir, modality, dataset_name + '_' + 'raw_training_scores_{}_{}.npy'.format(foreground_extraction_mode, method)))
        torch.save(of_training_scores_set, os.path.join(data_root_dir, modality, dataset_name + '_' + 'of_training_scores_{}_{}.npy'.format(foreground_extraction_mode, method)))
    else:
        raw_losses = AverageMeter()
        of_losses = AverageMeter()
        for h_idx in range(len(foreground_set)):
            for w_idx in range(len(foreground_set[h_idx])):
                cur_training_data = foreground_set[h_idx][w_idx]
                if len(cur_training_data) > 1:  # num > 1 for data parallel
                    cur_training_data2 = foreground_set2[h_idx][w_idx]
                    cur_dataset = cube_to_train_dataset(cur_training_data, target=cur_training_data2)
                    cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=True)

                    cur_model = torch.nn.DataParallel(network_architecture).cuda()
                    optimizer = optim.Adam(cur_model.parameters(), eps=1e-7, weight_decay=0.0)
                    cur_model.train()
                    for epoch in range(epochs):
                        for idx, (inputs, of_targets_all, _) in enumerate(cur_dataloader):
                            inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                            of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)

                            of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)

                            loss_raw = loss_func(raw_targets.detach(), raw_outputs)
                            if useFlow:
                                loss_of = loss_func(of_targets.detach(), of_outputs)

                            if useFlow:
                                loss = lambda_raw * loss_raw + lambda_of * loss_of
                            else:
                                loss = loss_raw

                            raw_losses.update(loss_raw.item(), inputs.size(0))
                            if useFlow:
                                of_losses.update(loss_of.item(), inputs.size(0))
                            else:
                                of_losses.update(0., inputs.size(0))

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            if idx % 5 == 0:
                                print('Block: ({}, {}), epoch {}, batch {} of {}, raw loss: {}, of loss: {}'.format(h_idx, w_idx, epoch, idx,
                                                                                                                    cur_dataset.__len__() // batch_size,
                                                                                                                    raw_losses.avg,
                                                                                                                    of_losses.avg))

                    model_set[h_idx][w_idx].append(cur_model.state_dict())

                    # A forward pass to store the training scores of optical flow and raw datasets respectively.
                    forward_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=False)
                    score_func = nn.MSELoss(reduce=False)
                    cur_model.eval()
                    for idx, (inputs, of_targets_all, _) in enumerate(forward_dataloader):
                        inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                        of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)
                        of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)

                        raw_scores = score_func(raw_targets, raw_outputs).cpu().data.numpy()
                        raw_scores = np.sum(np.sum(np.sum(raw_scores, axis=3), axis=2), axis=1)  # mse
                        raw_training_scores_set[h_idx][w_idx].append(raw_scores)
                        if useFlow:
                            of_scores = score_func(of_targets, of_outputs).cpu().data.numpy()
                            of_scores = np.sum(np.sum(np.sum(of_scores, axis=3), axis=2), axis=1)  # mse
                            of_training_scores_set[h_idx][w_idx].append(of_scores)

                    raw_training_scores_set[h_idx][w_idx] = np.concatenate(raw_training_scores_set[h_idx][w_idx], axis=0)
                    if useFlow:
                        of_training_scores_set[h_idx][w_idx] = np.concatenate(of_training_scores_set[h_idx][w_idx], axis=0)
        torch.save(raw_training_scores_set, os.path.join(data_root_dir, modality, dataset_name + '_' + 'raw_training_scores_{}_{}.npy'.format(foreground_extraction_mode, method)))
        torch.save(of_training_scores_set, os.path.join(data_root_dir, modality, dataset_name + '_' + 'of_training_scores_{}_{}.npy'.format(foreground_extraction_mode, method)))
        print('training scores saved!')

    torch.save(model_set, os.path.join(data_root_dir, modality, dataset_name+'_'+'model_{}_{}.npy'.format(foreground_extraction_mode, method)))
    print('Training of {} for dataset: {} has completed!'.format(method, dataset_name))
else:
    raise NotImplementedError

