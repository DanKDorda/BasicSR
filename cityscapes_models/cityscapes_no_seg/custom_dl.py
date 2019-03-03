"""
dataloader for reading the cityscapes dataset HR images and segmentations, and performing on-line downsampling
"""
import cv2
import random

import numpy as np
import torch
import torch.utils.data as data
from enum import Enum
import codes.data.util as util


class data_folders(Enum):
    cs = 'cityscapes'
    ot = 'overfit_train'

def build_custom_dataloader(opt, phase):
    opt['HR_size'] = opt['datasets']['train']['HR_size']
    dataset = Cityscape_SR_dataset(opt, phase)
    dl_opt = opt['datasets']['train']
    return data.DataLoader(
        dataset,
        batch_size=dl_opt['batch_size'],
        shuffle=dl_opt['use_shuffle'],
        num_workers=dl_opt['n_workers'],
        drop_last=True,
        pin_memory=True)


class Cityscape_SR_dataset(data.Dataset):

    def __init__(self, opt, phase):
        super(Cityscape_SR_dataset, self).__init__()
        self.opt = opt
        self.phase = phase
        # Todo: use the enum >:[
        dir_hr = f'/home/daniel/storage/BasicSR/data_samples/overfit_train/{phase}/hr'
        dir_lr = f'/home/daniel/storage/BasicSR/data_samples/overfit_train/{phase}/lr'
        dir_seg = f'/home/daniel/storage/BasicSR/data_samples/overfit_train/{phase}/seg'
        # self.paths_LR = util.get_image_paths('img', dir_lr)
        self.paths_LR = None
        self.paths_HR = util.get_image_paths('img', dir_hr)
        self.paths_seg = util.get_image_paths('img', dir_seg)
        ##### HACK ALERT
        # TODO: fix this "feature"
        self.paths_HR = self.paths_HR[1]
        self.paths_seg = self.paths_seg[1]

        # get hr paths
        # get lr paths??? how to handle multi-scale?? oh wait, SFTGAN don't use multiscale, there's just one x4 folder

        # run assertions
        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format( \
                    len(self.paths_LR), len(self.paths_HR))

        assert len(self.paths_HR) == len(self.paths_seg), \
            'HR and seg datasets have different number of images - {}, {}.'.format( \
                len(self.paths_LR), len(self.paths_HR))

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']

        # read HR image and segmentation
        HR_path = self.paths_HR[index]
        # print(HR_path)
        img_HR = util.read_img(None, HR_path)
        seg_path = self.paths_seg[index]
        # print(f'seg path:{seg_path}')
        seg = util.read_img(None, seg_path)
        # print(f'seg size:{seg.shape}')

        # get LR image
        if False and self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = util.read_img(self.LR_env, LR_path)
        else:  # down-sampling on-the-fly

            H, W, _ = img_HR.shape
            H_s, W_s = H // scale, W // scale
            # using matlab imresize
            img_LR = util.imresize_np(img_HR, 1 / scale, True)
            # seg = util.imresize_np(seg, 1 / scale, True)
            # seg = cv2.resize(np.copy(seg), (W_s, H_s), interpolation=cv2.INTER_NEAREST)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        # print(f'seg size:{seg.shape}')
        # seg = np.expand_dims(seg, 2)
        # data augmentation in training
        # TODO: fix hakk
        # seg = seg[:, :, 0]

        if self.phase == 'train':
            H, W, C = img_LR.shape
            LR_size = HR_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]
            seg = seg[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size]

            # augmentation - flip, rotate
            img_LR, img_HR, seg = util.augment([img_LR, img_HR, seg], self.opt['use_flip'],
                                               self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        seg = torch.from_numpy(np.ascontiguousarray(np.transpose(seg, (2, 0, 1)))).float()
        seg = self.seg_to_onehot(seg)

        if LR_path is None:
            LR_path = HR_path

        return {
            'LR': img_LR,
            'HR': img_HR,
            'seg': seg,
            'LR_path': LR_path,
            'HR_path': HR_path
        }

    def __len__(self):
        return len(self.paths_HR)

    def seg_to_onehot(self, seg):
        size = seg.size()
        seg = seg*255
        oneHot_size = (self.opt['n_labels'], size[1], size[2])
        input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(0, seg.data.long(), 255.0)
        return input_label


if __name__ == '__main__':
    import argparse
    import codes.options.options as option
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='/home/daniel/storage/BasicSR/codes/options/train/train_cityscapeSFT.json',
                        help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt['scale'] = 4
    opt['HR_size'] = 512

    phase = 'train'
    ds = Cityscape_SR_dataset(opt, phase)
    data0 = ds[20]
    im_hr = data0['HR']
    im_lr = data0['LR']
    im_seg = data0['seg']
    print(im_hr.shape)
    print(im_seg.shape)

    plt.subplot(3, 1, 1)
    plt.imshow(im_hr.permute(1, 2, 0).numpy())
    plt.subplot(3, 1, 2)
    plt.imshow(im_lr.permute(1, 2, 0).numpy())
    plt.subplot(3, 1, 3)
    print(im_seg.device)
    im_seg = torch.argmax(im_seg, 0)
    print(im_seg.shape)
    plt.imshow(im_seg.numpy())

    plt.show()
