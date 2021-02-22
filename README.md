## [Cloze Test Helps: Effective Video Anomaly Detection via Learning to Complete Video Events](https://www.researchgate.net/publication/343809709_Cloze_Test_Helps_Effective_Video_Anomaly_Detection_via_Learning_to_Complete_Video_Events)

This repository is the official implementation of [Cloze Test Helps: Effective Video Anomaly Detection via Learning to Complete Video Events](https://dl.acm.org/doi/10.1145/3394171.3413973) (oral paper In ACM Multimedia 2020) by Guang Yu, Siqi Wang, Zhiping Cai, En Zhu, Chuanfu Xu, Jianping Yin, Marius Kloft. 

## 1. Environment

* python 3.6
* PyTorch 1.1.0 (0.3.0 for calculating optical flow)
* torchvision 0.3.0
* cuda 9.0.176
* cudnn 7.0.5
* mmcv 0.2.14 (might use `pip install mmcv==0.2.14` to install old version)
* [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v1.0rc0) 1.0rc0 (might use `git clone -b v1.0rc0 https://github.com/open-mmlab/mmdetection.git` to clone old version)
* numpy 1.17.2
* scikit-learn 0.21.3

Refer to the full environment in [issue](https://github.com/yuguangnudt/VEC_VAD/issues/2).  Note that our project is based on mmdet v1.0rc0. Run the program strictly according to our environment, or might try the newer versions of mmdet, PyTorch and mmcv.

Recently (2021.1) the interface of mmdet v1.0rc0 seems to have changed. If you install mmdet v1.0rc0 and get "No module named 'mmdet.datasets.pipelines' " when running the program, please refer to [issue](https://github.com/yuguangnudt/VEC_VAD/issues/9#issuecomment-768020917) to fix the bug.

## 2. Download and organize datasets

Download UCSDped2 from [official website](http://svcl.ucsd.edu/projects/anomaly/dataset.htm) , avenue and Shanghaitech from [OneDrive](https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F) or [BaiduYunPan](https://pan.baidu.com/s/1j0TEt-2Dw3kcfdX-LCF0YQ) (code:i9b3, provided by [StevenLiuWen](https://github.com/StevenLiuWen/ano_pred_cvpr2018)) , and [ground truth](www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip) of avenue from [official website](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html). Create a folder named `raw_datasets` in root directory to store the downloaded datasets. The directory structure should be organized to match `vad_datasets.py` as follows (Refer to the entire project directory structure in `directory_structure.txt`): 

```
.
├── ...
├── raw_datasets
 │   ├── avenue
 │   │   ├── bboxes_test_obj_det_with_motion.npy
 │   │   ├── bboxes_train_obj_det_with_motion.npy
 │   │   ├── ground_truth_demo
 │   │   ├── testing
 │   │   └── training
 │   ├── ShanghaiTech
 │   │   ├── bboxes_test_obj_det_with_motion.npy
 │   │   ├── bboxes_train_obj_det_with_motion.npy
 │   │   ├── extract_frames.py
 │   │   ├── Testing
 │   │   ├── training
 │   │   └── training.zip
 │   ├── UCSDped2
 │   │   ├── bboxes_test_obj_det_with_motion.npy
 │   │   ├── bboxes_train_obj_det_with_motion.npy
 │   │   ├── Test
 │   │   └── Train
├── calc_optical_flow.py
├── ...
```

**Note:** (1) To facilitate testing and training, extracted foreground bounding boxes (`bboxes_test_obj_det_with_motion.npy`, `bboxes_train_obj_det_with_motion.npy`)  have been uploaded to the directories of each dataset. Please set `train_bbox_saved=True` and `test_bbox_saved=True`  in `config.cfg` to load the extracted bboxes directly if you don't want to extract bboxes using mmdet. (2) ShanghaiTech's training set provides videos rather than video frames, which need to be extracted manually. `extract_frames.py` have been uploaded to `./raw_datasets/ShanghaiTech` for video frame extraction. After downloading and unzipping ShanghaiTech, run `extract_frames.py` to get the video frames of ShanghaiTech training set.

## 3. Calculate optical flow

(1) Follow the [instructions](https://github.com/vt-vl-lab/flownet2.pytorch) to install FlowNet2, then download the pretrained model  [flownet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing), and move the downloaded model `FlowNet2_checkpoint.pth.tar` into `./FlowNet2_src/pretrained` (create a folder named pretrained).

(2) Run `calc_optical_flow.py` (in PyTorch 0.3.0): `python calc_optical_flow.py`. This will generate a new folder named `optical_flow` containing the optical flow of the different datasets. The `optical_flow` folder has basically the same directory structure as the raw_datasets folder.

## 4.  Test on saved models

(1) Follow the [instructions](https://github.com/open-mmlab/mmdetection/tree/v1.0rc0) to install mmdet (might use `git clone -b v1.0rc0 https://github.com/open-mmlab/mmdetection.git` to clone old version of mmdetection). Then download the pretrained object detector [Cascade R-CNN](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth), and move it to `fore_det/obj_det_checkpoints` (create a folder named obj_det_checkpoints).

(2) Select the model in `./data/raw2flow`, and move the files in the model folder (such as `avenue_model_5raw1of_auc0.902`) into `./data/raw2flow`. 

(3) Edit the file `config.cfg`: i. Set the `dataset_name` (`UCSDped2`,  `avenue` and `ShanghaiTech` are optional) of `[shared_parameters]` for the selected model in  step (2).  ii. Set the `context_of_num` (4 and 0 are optional, 4 corresponds to the model with `5of` and 0 corresponds to `1of`) in `[SelfComplete]`.

(4) Run `test.py`: `python test.py`.

## 5. Train

Edit the file `config.cfg` according to Experimental Settings in our paper or your requirements, and run `train.py`: `python train.py`.

## 6. Performance

| Dataset | UCSDped2 | avenue | ShanghaiTech |
| :-----: | :------: | :----: | :----------: |
|  AUROC  |  97.3%   | 90.2%  |    74.8%     |

Extensions and higher performance will be released!

## 7. Citation

```
@inproceedings{yu2020cloze,
  title={Cloze Test Helps: Effective Video Anomaly Detection via Learning to Complete  	  Video Events},
  author={Yu, Guang and Wang, Siqi and Cai, Zhiping and Zhu, En and Xu, Chuanfu and Yin, Jianping and Kloft, Marius},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={583--591},
  year={2020}
}
```





