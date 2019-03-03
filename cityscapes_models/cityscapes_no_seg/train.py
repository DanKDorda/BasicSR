"""
train an SFT SRGAN on cityscapes data
"""
import argparse
import os
from tensorboardX import SummaryWriter
import torch
import time
import random

# print(sys.path)

from codes.options import options as option
from cityscapes_models.cityscapes_no_seg import custom_dl as cdl
from codes.models import create_model
from codes.utils import util


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='/home/daniel/storage/BasicSR/codes/options/train/train_cityscapeSFT.json',
                        help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    print(opt)

    # TODO: check if this folder setting code works
    if opt['path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['path']['resume_state'])
    else:  # training from scratch
        resume_state = None
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # TODO: deal with the use of "logging" in this crackden
    writer = SummaryWriter()
    writer.add_text('Options', str(opt))

    torch.backends.cudnn.benchmark = True

    # create train and val dataloader
    train_loader = cdl.build_custom_dataloader(opt, 'train')
    val_loader = cdl.build_custom_dataloader(opt, 'val')

    model = create_model(opt)

    # training
    current_step = 0
    start_epoch = 0
    total_epochs = 2000
    epoch_len = len(train_loader)

    print('starting training')
    for epoch in range(start_epoch, total_epochs):
        tic = time.time()
        for _, train_data in enumerate(train_loader):
            current_step += 1

            # model.update_learning_rate()

            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if current_step % 30 == 1:
                losses = model.get_current_losses()
                epoch_step = current_step % epoch_len
                visuals = model.get_current_visuals()
                gt_img = util.tensor2img(visuals['HR'])  # uint8
                sr_img = util.tensor2img(visuals['SR'])  # uint8

                writer.add_image('GT_train', gt_img, current_step, dataformats='HWC')
                writer.add_image('SR_train', sr_img, current_step, dataformats='HWC')

                print(f'step {epoch_step}/{epoch_len} || loss: {losses}')
                writer.add_scalars('Losses', losses, current_step)

            if current_step % opt['train']['val_freq'] == 0:
                print('validating model')
                avg_psnr = validate(current_step, model, opt, val_loader)
                writer.add_scalar('Validation loss', avg_psnr, current_step)

            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                print('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

            # more logging
        if epoch % 10 == 0:
            print(f'epoch {epoch} done, took {time.time()-tic}s')


def validate(current_step, model, opt, val_loader):
    torch.backends.cudnn.benchmark = False

    avg_psnr = 0.0
    idx = 0
    do_save = True
    for val_data in val_loader:
        idx += 1
        img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
        img_dir = os.path.join(opt['path']['val_images'], img_name)
        util.mkdir(img_dir)

        model.feed_data(val_data)
        model.test()

        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals['SR'])  # uint8
        gt_img = util.tensor2img(visuals['HR'])  # uint8

        # Save SR images for reference
        if do_save:
            save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format( \
                img_name, current_step))
            util.save_img(sr_img, save_img_path)

        do_save = random.random() < 0.05

        # calculate PSNR
        crop_size = opt['scale']
        gt_img = gt_img / 255.
        sr_img = sr_img / 255.
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
        avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
    avg_psnr = avg_psnr / idx

    torch.backends.cudnn.benchmark = True

    return avg_psnr


if __name__ == '__main__':
    main()
