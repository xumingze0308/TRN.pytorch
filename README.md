# Temporal Recurrent Networks for Online Action Detection

## Updates

:boom: **May 25th 2021**: For future comparison with TRN using Kinetics pretrained features, we report our results on THUMOS as 62.1%, on TVSeries as 86.2%, and on HACS Segment as 78.9%, in terms of mAP.

## Introduction

This is a PyTorch **reimplementation** for our ICCV 2019 paper "[`Temporal Recurrent Networks for Online Action Detection`](https://arxiv.org/pdf/1811.07391.pdf)".

![network](demo/network.jpg?raw=true)

## Environment

- The code is developed with CUDA 9.0, ***Python >= 3.6***, ***PyTorch >= 1.0***

## Data Preparation

1. Download the [`HDD`](https://usa.honda-ri.com/hdd) and [`THUMOS'14`](https://www.crcv.ucf.edu/THUMOS14/) datasets.

2. Extract feature representations for video frames.

    * For HDD dataset, we use the [`Inception-ResNet-V2`](https://arxiv.org/pdf/1602.07261.pdf) pretrained on ImageNet for the RGB input.
    
    * For THUMOS'14 dataset, we use the [`ResNet-200`](https://arxiv.org/pdf/1512.03385.pdf) model for the RGB input, and the [`BN-Inception`](https://arxiv.org/pdf/1502.03167.pdf) model for the optical flow input. To replicate our results, please follow the repo here: [`https://github.com/yjxiong/anet2016-cuhk`](https://github.com/yjxiong/anet2016-cuhk).
    
    ***Note:*** We compute the optical flow for the THUMOS'14 dataset using [`FlowNet2.0`](https://arxiv.org/pdf/1612.01925.pdf).

3. If you want to use our [dataloaders](./lib/datasets), please make sure to put the files as the following structure:

    * HDD dataset:
    ```
    $YOUR_PATH_TO_HDD_DATASET
    ├── inceptionresnetv2/
    |   ├── 201702271017.npy (of size L x 1536 x 8 x 8)
    │   ├── ...
    ├── sensor/
    |   ├── 201702271017.npy (of size L x 8)
    |   ├── ...
    ├── target/
    |   ├── 201702271017.npy (of size L)
    |   ├── ...
    ```
    
    * THUMOS'14 dataset:
    ```
    $YOUR_PATH_TO_THUMOS_DATASET
    ├── resnet200-fc/
    |   ├── video_validation_0000051.npy (of size L x 2048)
    │   ├── ...
    ├── bn_inception/
    |   ├── video_validation_0000051.npy (of size L x 1024)
    |   ├── ...
    ├── target/
    |   ├── video_validation_0000051.npy (of size L x 22)
    |   ├── ...
    ```

4. Create softlinks of datasets:
    ```
    cd TRN.pytorch
    ln -s $YOUR_PATH_TO_HDD_DATASET data/HDD
    ln -s $YOUR_PATH_TO_THUMOS_DATASET data/THUMOS
    ```

## Training

* Single GPU training on HDD dataset:
```
cd TRN.pytorch
# Training from scratch
python tools/trn_hdd/train.py --gpu $CUDA_VISIBLE_DEVICES
# Finetuning from a pretrained model
python tools/trn_hdd/train.py --checkpoint $PATH_TO_CHECKPOINT --gpu $CUDA_VISIBLE_DEVICES
```

* Multi-GPU training on HDD dataset:
```
cd TRN.pytorch
# Training from scratch
python tools/trn_hdd/train.py --gpu $CUDA_VISIBLE_DEVICES --distributed
# Finetuning from a pretrained model
python tools/trn_hdd/train.py --checkpoint $PATH_TO_CHECKPOINT --gpu $CUDA_VISIBLE_DEVICES --distributed
```

* Single GPU training on THUMOS'14 dataset:
```
cd TRN.pytorch
# Training from scratch
python tools/trn_thumos/train.py --gpu $CUDA_VISIBLE_DEVICES
# Finetuning from a pretrained model
python tools/trn_thumos/train.py --checkpoint $PATH_TO_CHECKPOINT --gpu $CUDA_VISIBLE_DEVICES
```

* Multi-GPU training on THUMOS'14 dataset:
```
cd TRN.pytorch
# Training from scratch
python tools/trn_thumos/train.py --gpu $CUDA_VISIBLE_DEVICES --distributed
# Finetuning from a pretrained model
python tools/trn_thumos/train.py --checkpoint $PATH_TO_CHECKPOINT --gpu $CUDA_VISIBLE_DEVICES --distributed
```

## Evaluation

* HDD dataset:
```
cd TRN.pytorch
python tools/trn_hdd/eval.py --checkpoint $PATH_TO_CHECKPOINT --gpu $CUDA_VISIBLE_DEVICES
```

* THUMOS'14 dataset:
```
cd TRN.pytorch
python tools/trn_thumos/eval.py --checkpoint $PATH_TO_CHECKPOINT --gpu $CUDA_VISIBLE_DEVICES
```

***NOTE:*** There are two kinds of evaluation methods in our code. (1) Using `--debug` during training considers each short video clip (consisting of 90 and 64 consecutive frames for HDD and THUMOS'14 datasets, respectively) as one test sample, and separately runs inference and evaluates on all short video clips (even though some of them are from the same long video). (2) Using `eval.py` after training runs inference and evaluates on long videos (frame by frame, from the beginning to the end), which is the evaluation method we reported in the paper.

## Citations

If you are using the data/code/model provided here in a publication, please cite our paper:

    @inproceedings{onlineaction2019iccv,
        title = {Temporal Recurrent Networks for Online Action Detection},
        author = {Mingze Xu and Mingfei Gao and Yi-Ting Chen and Larry S. Davis and David J. Crandall},
        booktitle = {IEEE International Conference on Computer Vision (ICCV)},
        year = {2019}
    }
