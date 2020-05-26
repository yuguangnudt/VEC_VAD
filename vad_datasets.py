import torch
import numpy as np
import cv2
from collections import OrderedDict
import os
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
        transforms.ToTensor(),
    ])
# frame_size: the frame information of each dataset: (h, w, file_format, scene_num)
frame_size = {'UCSDped1' : (158, 238, '.tif', 1), 'UCSDped2': (240, 360, '.tif', 1), 'avenue': (360, 640, '.jpg', 1), 'ShanghaiTech': (480, 856, '.jpg', 1)}

def get_inputs(file_addr):
    file_format = file_addr.split('.')[-1]
    if file_format == 'mat':
        return sio.loadmat(file_addr, verify_compressed_data_integrity=False)['uv']
    elif file_format == 'npy':
        return np.load(file_addr)
    else:
        return cv2.imread(file_addr)

def img_tensor2numpy(img):
    # mutual transformation between ndarray-like imgs and Tensor-like images
    # both intensity and rgb images are represented by 3-dim data
    if isinstance(img, np.ndarray):
        return torch.from_numpy(np.transpose(img, [2, 0, 1]))
    else:
        return np.transpose(img, [1, 2, 0]).numpy()

def img_batch_tensor2numpy(img_batch):
    # both intensity and rgb image batch are represented by 4-dim data
    if isinstance(img_batch, np.ndarray):
        if len(img_batch.shape) == 4:
            return torch.from_numpy(np.transpose(img_batch, [0, 3, 1, 2]))
        else:
            return torch.from_numpy(np.transpose(img_batch, [0, 1, 4, 2, 3]))
    else:
        if len(img_batch.numpy().shape) == 4:
            return np.transpose(img_batch, [0, 2, 3, 1]).numpy()
        else:
            return np.transpose(img_batch, [0, 1, 3, 4, 2]).numpy()

class bbox_collate:
    def __init__(self, mode):
        self.mode = mode

    def collate(self, batch):
        if self.mode == 'train':
            return bbox_collate_train(batch)
        elif self.mode == 'test':
            return bbox_collate_test(batch)
        else:
            raise NotImplementedError

def bbox_collate_train(batch):
    batch_data = [x[0] for x in batch]
    batch_target = [x[1] for x in batch]
    return torch.cat(batch_data, dim=0), batch_target

def bbox_collate_test(batch):
    batch_data = [x[0] for x in batch]
    batch_target = [x[1] for x in batch]
    return batch_data, batch_target

def get_foreground(img, bboxes, patch_size):
    img_patches = list()
    if len(img.shape) == 3:
        for i in range(len(bboxes)):
            x_min, x_max = np.int(np.ceil(bboxes[i][0])), np.int(np.ceil(bboxes[i][2]))
            y_min, y_max = np.int(np.ceil(bboxes[i][1])), np.int(np.ceil(bboxes[i][3]))
            cur_patch = img[:, y_min:y_max, x_min:x_max]
            cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]), (patch_size, patch_size))
            img_patches.append(np.transpose(cur_patch, [2, 0, 1]))
        img_patches = np.array(img_patches)
    elif len(img.shape) == 4:
        for i in range(len(bboxes)):
            x_min, x_max = np.int(np.ceil(bboxes[i][0])), np.int(np.ceil(bboxes[i][2]))
            y_min, y_max = np.int(np.ceil(bboxes[i][1])), np.int(np.ceil(bboxes[i][3]))
            cur_patch_set = img[:, :, y_min:y_max, x_min:x_max]
            tmp_set = list()
            for j in range(img.shape[0]):
                cur_patch = cur_patch_set[j]
                cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]), (patch_size, patch_size))
                tmp_set.append(np.transpose(cur_patch, [2, 0, 1]))
            cur_cube = np.array(tmp_set)
            img_patches.append(cur_cube)
        img_patches = np.array(img_patches)
    return img_patches

def unified_dataset_interface(dataset_name, dir, mode='train', context_frame_num=0, border_mode='elastic', file_format=None, all_bboxes=None, patch_size=32):

    if file_format is None:
        if dataset_name in ['UCSDped1', 'UCSDped2']:
            file_format = '.tif'
        elif dataset_name in ['avenue', 'ShanghaiTech']:
            file_format = '.jpg'
        else:
            raise NotImplementedError

    if dataset_name in ['UCSDped1', 'UCSDped2']:
        dataset = ped_dataset(dir=dir, context_frame_num=context_frame_num, mode=mode, border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format)
    elif dataset_name == 'avenue':
        dataset = avenue_dataset(dir=dir, context_frame_num=context_frame_num, mode=mode, border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format)
    elif dataset_name == 'ShanghaiTech':
        dataset = shanghaiTech_dataset(dir=dir, context_frame_num=context_frame_num, mode=mode, border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format)
    else:
        raise NotImplementedError

    return dataset

class patch_to_train_dataset(Dataset):
    def __init__(self, data, tranform=transform):
        self.data = data
        self.transform = tranform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, indice):
        if self.transform is not None:
            return self.transform(self.data[indice])
        else:
            return self.data[indice]

class cube_to_train_dataset(Dataset):
    def __init__(self, data, target=None, tranform=transform):
        if len(data.shape) == 4:
            data = data[:, np.newaxis, :, :, :]
        if len(target.shape) == 4:
            target = target[:, np.newaxis, :, :, :]
        self.data = data
        self.target = target
        self.transform = tranform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, indice):
        if self.target is None:
            cur_data = self.data[indice]
            cur_train_data = cur_data[:-1]
            cur_target = cur_data[-1]
            cur_train_data = np.transpose(cur_train_data, [1, 2, 0, 3])
            cur_train_data = np.reshape(cur_train_data, (cur_train_data.shape[0], cur_train_data.shape[1], -1))
            if self.transform is not None:
                return self.transform(cur_train_data), self.transform(cur_target)
            else:
                return cur_train_data, cur_target
        else:
            cur_data = self.data[indice]
            cur_train_data = cur_data
            cur_target = self.target[indice]
            cur_target2 = cur_data.copy()
            cur_train_data = np.transpose(cur_train_data, [1, 2, 0, 3])
            cur_train_data = np.reshape(cur_train_data, (cur_train_data.shape[0], cur_train_data.shape[1], -1))
            cur_target = np.transpose(cur_target, [1, 2, 0, 3])
            cur_target = np.reshape(cur_target, (cur_target.shape[0], cur_target.shape[1], -1))
            cur_target2 = np.transpose(cur_target2, [1, 2, 0, 3])
            cur_target2 = np.reshape(cur_target2, (cur_target2.shape[0], cur_target2.shape[1], -1))
            if self.transform is not None:
                return self.transform(cur_train_data), self.transform(cur_target), self.transform(cur_target2)
            else:
                return cur_train_data, cur_target, cur_target2

class ped_dataset(Dataset):
    '''
    Loading dataset for UCSD ped2
    '''
    def __init__(self, dir, mode='train', context_frame_num=0, border_mode='elastic', file_format='.tif', all_bboxes=None, patch_size=32):
        '''
        :param dir: The directory to load UCSD ped2 dataset
        mode: train/test dataset
        '''
        self.dir = dir
        self.mode = mode
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.frame_video_idx = list()
        self.tot_frame_num = 0
        self.context_frame_num = context_frame_num
        self.border_mode = border_mode
        self.file_format = file_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size
        self.return_gt = False
        if mode == 'test':
            self.all_gt_addr = list()
            self.gts = OrderedDict()
        if self.dir[-1] == '1':
            self.h = 158
            self.w = 238
        else:
            self.h = 240
            self.w = 360
        self.dataset_init()

    def __len__(self):
        return self.tot_frame_num

    def dataset_init(self):
        if self.mode == 'train':
            data_dir = os.path.join(self.dir, 'Train')
        elif self.mode == 'test':
            data_dir = os.path.join(self.dir, 'Test')
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                if 'Train' in video_name:
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                    self.videos[video_name]['frame'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

        elif self.mode == 'test':
            dir_list = glob.glob(os.path.join(data_dir, '*'))
            video_dir_list = []
            gt_dir_list = []
            for dir in sorted(dir_list):
                if '_gt' in dir:
                    gt_dir_list.append(dir)
                    self.return_gt = True
                else:
                    name = dir.split('/')[-1]
                    if 'Test' in name:
                        video_dir_list.append(dir)

            # load frames for test
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # load ground truth of frames
            if self.return_gt:
                for gt in sorted(gt_dir_list):
                    gt_name = gt.split('/')[-1]
                    self.gts[gt_name] = {}
                    self.gts[gt_name]['gt_frame'] = glob.glob(os.path.join(gt, '*.bmp'))
                    self.gts[gt_name]['gt_frame'].sort()

                # merge different frames of different videos into one list
                for _, cont in self.gts.items():
                    self.all_gt_addr += cont['gt_frame']

        else:
            raise NotImplementedError

    def context_range(self, indice):
        if self.border_mode == 'elastic':
            # check head and tail
            if indice - self.context_frame_num < 0:
                indice = self.context_frame_num
            elif indice + self.context_frame_num > self.tot_frame_num - 1:
                indice = self.tot_frame_num - 1 - self.context_frame_num
            start_idx = indice - self.context_frame_num
            end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1
        elif self.border_mode == 'predict':
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num
            end_idx = indice
            need_context_num = self.context_frame_num + 1
        else:
            # check head and tail
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num

            if indice + self.context_frame_num > self.tot_frame_num - 1:
                end_idx = self.tot_frame_num - 1
            else:
                end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1

        center_idx = self.frame_video_idx[indice]
        video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        pad = need_context_num - len(video_idx)
        if pad > 0:
            if start_idx == 0:
                video_idx = [video_idx[0]] * pad + video_idx
            else:
                video_idx = video_idx + [video_idx[-1]] * pad
        tmp = np.array(video_idx) - center_idx
        offset = tmp.sum()
        if tmp[0] != 0 and tmp[-1] != 0:  # extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError
        if pad == 0 and offset == 0:  # all frames are from the same video
            idx = [x for x in range(start_idx, end_idx+1)]
            return idx
        else:
            if self.border_mode == 'elastic':
                idx = [x for x in range(start_idx - offset, end_idx - offset + 1)]
                return idx
            elif self.border_mode == 'predict':
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), pad) + idx
                return idx
            else:
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if pad > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx - offset, end_idx + 1)]
                        idx = [idx[0]] * pad + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx - offset + 1)]
                        idx = idx + [idx[-1]] * pad
                        return idx

    def __getitem__(self, indice):

        if self.mode == 'train':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice]), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            return img_batch, torch.zeros(1)  # to unify the interface
        elif self.mode == 'test':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice]), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = cv2.imread(self.all_gt_addr[indice], cv2.IMREAD_GRAYSCALE)
                    gt_batch = torch.from_numpy(gt_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = cv2.imread(self.all_gt_addr[indice], cv2.IMREAD_GRAYSCALE)
                    gt_batch = torch.from_numpy(gt_batch)
            if self.return_gt:
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)  # to unify the interface
        else:
            raise NotImplementedError

class avenue_dataset(Dataset):
    '''
    Loading dataset for Avenue
    '''
    def __init__(self, dir, mode='train', context_frame_num=0, border_mode='elastic', file_format='.jpg', all_bboxes=None, patch_size=32):
        '''
        :param dir: The directory to load Avenue dataset
        mode: train/test dataset
        '''
        self.dir = dir
        self.mode = mode
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.frame_video_idx = list()
        self.tot_frame_num = 0
        self.context_frame_num = context_frame_num
        self.border_mode = border_mode
        self.file_format = file_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size
        self.return_gt = False
        if mode == 'test':
            self.all_gt = list()
        self.dataset_init()
        pass

    def __len__(self):
        return self.tot_frame_num

    def dataset_init(self):
        if self.mode == 'train':
            data_dir = os.path.join(self.dir, 'training', 'frames')
        elif self.mode == 'test':
            data_dir = os.path.join(self.dir, 'testing', 'frames')
            gt_dir = os.path.join(self.dir, 'ground_truth_demo', 'testing_label_mask')
            if os.path.exists(gt_dir):
                self.return_gt = True
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

        elif self.mode == 'test':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # set address of ground truth of frames
            if self.return_gt:
                self.all_gt = [sio.loadmat(os.path.join(gt_dir, str(x + 1)+'_label.mat'))['volLabel'] for x in range(len(self.videos))]
                self.all_gt = np.concatenate(self.all_gt, axis=1)
        else:
            raise NotImplementedError

    def context_range(self, indice):
        if self.border_mode == 'elastic':
            # check head and tail
            if indice - self.context_frame_num < 0:
                indice = self.context_frame_num
            elif indice + self.context_frame_num > self.tot_frame_num - 1:
                indice = self.tot_frame_num - 1 - self.context_frame_num
            start_idx = indice - self.context_frame_num
            end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1
        elif self.border_mode == 'predict':
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num
            end_idx = indice
            need_context_num = self.context_frame_num + 1
        else:
            # check head and tail
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num

            if indice + self.context_frame_num > self.tot_frame_num - 1:
                end_idx = self.tot_frame_num - 1
            else:
                end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1

        center_idx = self.frame_video_idx[indice]
        video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        pad = need_context_num - len(video_idx)
        if pad > 0:
            if start_idx == 0:
                video_idx = [video_idx[0]] * pad + video_idx
            else:
                video_idx = video_idx + [video_idx[-1]] * pad
        tmp = np.array(video_idx) - center_idx
        offset = tmp.sum()
        if tmp[0] != 0 and tmp[-1] != 0:  # extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError
        if pad == 0 and offset == 0:  # all frames are from the same video
            idx = [x for x in range(start_idx, end_idx+1)]
            return idx
        else:
            if self.border_mode == 'elastic':
                idx = [x for x in range(start_idx - offset, end_idx - offset + 1)]
                return idx
            elif self.border_mode == 'predict':
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), pad) + idx
                return idx
            else:
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if pad > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx - offset, end_idx + 1)]
                        idx = [idx[0]] * pad + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx - offset + 1)]
                        idx = idx + [idx[-1]] * pad
                        return idx

    def __getitem__(self, indice):
        if self.mode == 'train':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice]), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            return img_batch, torch.zeros(1)  # to unify the interface
        elif self.mode == 'test':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice]), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = self.all_gt[0, indice]
                    gt_batch = torch.from_numpy(gt_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = self.all_gt[0, indice]
                    gt_batch = torch.from_numpy(gt_batch)
            if self.return_gt:
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)
        else:
            raise NotImplementedError

class shanghaiTech_dataset(Dataset):
    '''
    Loading dataset for ShanghaiTech
    '''
    def __init__(self, dir, mode='train', context_frame_num=0, border_mode='elastic', file_format='.jpg', all_bboxes=None, patch_size=32):
        '''
        :param dir: The directory to load ShanghaiTech dataset
        mode: train/test dataset
        '''
        self.dir = dir
        self.mode = mode
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.frame_video_idx = list()
        self.tot_frame_num = 0
        self.context_frame_num = context_frame_num
        self.border_mode = border_mode
        self.file_format = file_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size
        self.return_gt = False
        self.save_scene_idx = list()
        self.scene_idx = list()
        self.scene_num = 0
        if mode == 'test':
            self.all_gt = list()
        self.dataset_init()
        pass

    def __len__(self):
        return self.tot_frame_num

    def dataset_init(self):
        if self.mode == 'train':
            data_dir = os.path.join(self.dir, 'training', 'videosFrame')
        elif self.mode == 'test':
            data_dir = os.path.join(self.dir, 'Testing', 'frames_part')
            gt_dir = os.path.join(self.dir, 'Testing', 'test_frame_mask')
            if os.path.exists(gt_dir):
                self.return_gt = True
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1
                self.save_scene_idx += [int(video_name[:2])] * len(self.videos[video_name]['frame'])  # frame data are saved by save_scene_idx
                self.scene_idx += [1] * len(self.videos[video_name]['frame'])  # framws are processed by scene idx

            self.scene_num = len(set(self.scene_idx))
            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

        elif self.mode == 'test':
            idx = 1
            for j in [1, 2]:
                video_dir_list = glob.glob(os.path.join(data_dir+str(j), '*'))
                for video in sorted(video_dir_list):
                    video_name = video.split('/')[-1]
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                    self.videos[video_name]['frame'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1
                    self.save_scene_idx += [int(video_name[:2])] * len(self.videos[video_name]['frame'])
                    self.scene_idx += [1] * len(self.videos[video_name]['frame'])

            self.scene_num = len(set(self.scene_idx))
            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # load ground truth of frames
            if self.return_gt:
                gt_dir_list = glob.glob(os.path.join(gt_dir, '*'))
                for gt in sorted(gt_dir_list):
                    self.all_gt.append(np.load(gt))

                # merge different frames of different videos into one list, only support frame gt now due to memory issue
                self.all_gt = np.concatenate(self.all_gt, axis=0)
        else:
            raise NotImplementedError


    def context_range(self, indice):
        if self.border_mode == 'elastic':
            # check head and tail
            if indice - self.context_frame_num < 0:
                indice = self.context_frame_num
            elif indice + self.context_frame_num > self.tot_frame_num - 1:
                indice = self.tot_frame_num - 1 - self.context_frame_num
            start_idx = indice - self.context_frame_num
            end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1
        elif self.border_mode == 'predict':
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num
            end_idx = indice
            need_context_num = self.context_frame_num + 1
        else:
            # check head and tail
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num

            if indice + self.context_frame_num > self.tot_frame_num - 1:
                end_idx = self.tot_frame_num - 1
            else:
                end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1

        center_idx = self.frame_video_idx[indice]
        video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        pad = need_context_num - len(video_idx)
        if pad > 0:
            if start_idx == 0:
                video_idx = [video_idx[0]] * pad + video_idx
            else:
                video_idx = video_idx + [video_idx[-1]] * pad
        tmp = np.array(video_idx) - center_idx
        offset = tmp.sum()
        if tmp[0] != 0 and tmp[-1] != 0:  # extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError
        if pad == 0 and offset == 0:  # all frames are from the same video
            idx = [x for x in range(start_idx, end_idx+1)]
            return idx
        else:
            if self.border_mode == 'elastic':
                idx = [x for x in range(start_idx - offset, end_idx - offset + 1)]
                return idx
            elif self.border_mode == 'predict':
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), pad) + idx
                return idx
            else:
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if pad > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx - offset, end_idx + 1)]
                        idx = [idx[0]] * pad + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx - offset + 1)]
                        idx = idx + [idx[-1]] * pad
                        return idx

    def __getitem__(self, indice):
        if self.mode == 'train':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice]), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            return img_batch, torch.zeros(1)  # to unify the interface
        elif self.mode == 'test':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice]), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = np.array([self.all_gt[indice]])
                    gt_batch = torch.from_numpy(gt_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx]), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = np.array([self.all_gt[indice]])
                    gt_batch = torch.from_numpy(gt_batch)
            if self.return_gt:
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)  # to unify the interface
        else:
            raise NotImplementedError

