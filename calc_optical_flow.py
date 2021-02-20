import torch
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
from vad_datasets import unified_dataset_interface
from FlowNet2_src import FlowNet2
from torch.autograd import Variable
from FlowNet2_src.flowlib import flow_to_image


def calc_optical_flow(dataset):
    of_root_dir = './optical_flow'
    len_original_root_dir = len(dataset.dir.split('/')) - 1
    flownet2 = FlowNet2()
    path = 'FlowNet2_src/pretrained/FlowNet2_checkpoint.pth.tar'
    pretrained_dict = torch.load(path)['state_dict']
    model_dict = flownet2.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    flownet2.load_state_dict(model_dict)
    flownet2.cuda()

    dataset_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    for idx, (batch, _) in enumerate(dataset_loader):
        print('Calculating optical flow for {}-th frame'.format(idx+1))
        cur_img_addr = dataset.all_frame_addr[idx]
        cur_img_name = cur_img_addr.split('/')[-1]
        cur_img_name = cur_img_name.split('.')[0]

        # parent path to store optical flow
        of_path = of_root_dir
        tmp_path_segment = cur_img_addr.split('/')[len_original_root_dir: -1]
        for cur_seg in tmp_path_segment:
            of_path = os.path.join(of_path, cur_seg)
        if os.path.exists(of_path) is False:
            os.makedirs(of_path, exist_ok=True)

        # Calculate optical flow by FlowNet2
        cur_imgs = np.transpose(batch[0].numpy(), [0, 2, 3, 1])
        frameRange = dataset.context_range(idx)
        old_size = (cur_imgs.shape[2], cur_imgs.shape[1])
        if frameRange[1] == frameRange[0] or frameRange[1] == frameRange[2]:
            if cur_imgs.shape[3] == 1:
                im1 = cv2.resize(cur_imgs[0], (512, 384))[:, :, np.newaxis]
                im2 = cv2.resize(cur_imgs[1], (512, 384))[:, :, np.newaxis]
                im1 = np.concatenate([im1] * 3, axis=2)
                im2 = np.concatenate([im2] * 3, axis=2)
            else:
                im1 = cv2.resize(cur_imgs[0], (512, 384))
                im2 = cv2.resize(cur_imgs[1], (512, 384))

            ims = np.array([[im1, im2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
            ims = torch.from_numpy(ims)

            ims_v = Variable(ims.cuda(), requires_grad=False)
            pred_flow = flownet2(ims_v).cpu().data
            pred_flow = pred_flow[0].numpy().transpose((1, 2, 0))
            new_inputs = cv2.resize(pred_flow, old_size)

        else:
            if cur_imgs.shape[3] == 1:
                im1 = cv2.resize(cur_imgs[1], (512, 384))[:, :, np.newaxis]
                im2 = cv2.resize(cur_imgs[2], (512, 384))[:, :, np.newaxis]
                im1 = np.concatenate([im1] * 3, axis=2)
                im2 = np.concatenate([im2] * 3, axis=2)
            else:
                im1 = cv2.resize(cur_imgs[1], (512, 384))
                im2 = cv2.resize(cur_imgs[2], (512, 384))

            ims = np.array([[im1, im2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
            ims = torch.from_numpy(ims)

            ims_v = Variable(ims.cuda(), requires_grad=False)
            pred_flow = flownet2(ims_v).cpu().data
            pred_flow = pred_flow[0].numpy().transpose((1, 2, 0))

            # Visualization
            # cv2.imshow('of', flow_to_image(pred_flow))
            # cv2.waitKey(0)

            new_inputs = cv2.resize(pred_flow, old_size)

        # Save optical flow
        np.save(os.path.join(of_path, cur_img_name+'.npy'), new_inputs)


if __name__ == '__main__':
    '''
        Calculate optical flow of different datasets. This will generate a new folder named optical_flow containing 
        the optical flow of the different datasets. The optical_flow folder has the same directory structure as the 
        raw_datasets folder.
        '''

    '''
    unified_dataset_interface
    Args:
        dataset_name: UCSDped2, avenue, ShanghaiTech
        mode: train, test (for calculating optical flow of training dataset and testing dataset respectively)
        other parameters: fixed
    Return:
        dataset
    '''

    # Example of calculating optical flow of UCSDped2 training and testing sets.
    dataset_name = 'UCSDped2'
    dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                        context_frame_num=1, mode='train', border_mode='hard')
    calc_optical_flow(dataset)
    dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                        context_frame_num=1, mode='test', border_mode='hard')
    calc_optical_flow(dataset)
    # The optical flow calculation of other datasets is basically the same as above.Change the dataset_name to calculate optical flow of other datasets.
