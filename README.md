## Cloze Test Helps: Effective Video Anomaly Detection via Learning to Complete Video Events

by anonymous author(s) for ACM Multimedia 2020. 

## 1. Environment

* python 3.6
* PyTorch 1.1.0 (0.3.0 for calculating optical flow)
* scikit-learn 0.21.3
* cuda 9.0.176
* cudnn 7.0

## 2. Download datasets

Download datasets from OneDrive (https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F) or BaiduYunPan (https://pan.baidu.com/s/1j0TEt-2Dw3kcfdX-LCF0YQ, code:i9b3) , and move them into `./raw_datasets`.

## 3. Calculate optical flow

(1) Download the pretrained model  flownet2 (https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing), and move the downloaded model `FlowNet2_checkpoint.pth.tar` into `./FlowNet2_src/pretrained`.

(2) Run `calc_img_inputs.py`: `python calc_img_inputs.py`. This will generate a new folder named `optical_flow` containing the optical flow of the different datasets.

For additional information, please refer to https://github.com/vt-vl-lab/pytorch_flownet2.

## 4.  Test on saved models

(1) Follow the instructions (https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md) to install mmdetection, then download the pretrained object detector Cascade R-CNN (https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth), and move it to `./obj_det_checkpoints`.

(2) Select the model in `./data/raw2flow`, and move the files in the folders (such as `avenue_model_5raw1of_auc0.902`) into `./data/raw2flow`. 

(3) Edit the file `config.cfg`: i. Change the `dataset_name` (`UCSDped2`,  `avenue` and `ShanghaiTech` are optional) for the model in `[shared_parameters]` .  ii. Change the `context_of_num` (4 and 0 are optional, 4 corresponds to the model with `5of` and 0 corresponds to `1of`) in `[SelfComplete]`.

(4) Run `test.py`: `python test.py`.

## 5. Train

Edit the file `config.cfg` according to your requirements and run `train.py`: `python train.py`.

## 6. Performance

| Dataset | UCSDped2 | Avenue | ShanghaiTech |
| :-----: | :------: | :----: | :----------: |
|  AUROC  |  97.3%   | 90.2%  |    74.8%     |
