import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from unet import VAE
from unet import EMA
from unet import GCALoss
from unet import GazeWeightedCrossEntropyLoss
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs_HMC-QU/')
dir_mask = Path('./data/masks_HMC-QU/')
dir_mask_gaze = Path('./data/HMC-QU_heatmap/')
dir_checkpoint = Path('./checkpoints/')

def train_model(
        model_teacher,
        model_stu,
        model_gca,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask,dir_mask_gaze,img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, dir_mask_gaze,img_scale)


    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    #experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    """
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )
    """
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    #ema = EMA(model_stu,model_teacher,0.999)
    #ema.register()
    params_list = nn.ModuleList([])
    params_list.append(model_stu)
    #params_list.append(model_gca)
    optimizer = optim.RMSprop(params_list.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    #optimizer = optim.RMSprop(list(model_stu.parameters()) + list(model_gca.parameters()),
                              #lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #criterion = nn.CrossEntropyLoss() if model_stu.n_classes > 1 else nn.BCEWithLogitsLoss()
    criterion = GazeWeightedCrossEntropyLoss()#眼动加权交叉熵loss
    criterion_heatmap = nn.MSELoss(reduction='mean')
    global_step = 0
    """
    optimizer_gca = optim.RMSprop(model_gca.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler_gca = optim.lr_scheduler.ReduceLROnPlateau(optimizer_gca, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler_gca = torch.cuda.amp.GradScaler(enabled=amp)
    criterion_gca = nn.CrossEntropyLoss() if model_stu.n_classes > 1 else nn.BCEWithLogitsLoss()
    """
    

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model_stu.train()
        model_teacher.train()
        model_gca.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks,heatmap = batch['image'], batch['mask'],batch['heatmap']

                assert images.shape[1] == model_stu.n_channels, \
                    f'Network has been defined with {model_stu.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                heatmap=heatmap.to(device=device, dtype=torch.float32)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred,heatmap_pred,x1_s,x2_s,x3_s,x4_s,x5_s = model_stu(images)
                    KV,heatmap_test,x1_t,x2_t,x3_t,x4_t,x5_t = model_teacher(images)
                    Q=heatmap
                    heatmap_pred=heatmap_pred.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    Q=Q.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    loss_gca =model_gca(x4_t,x4_s,Q)
                    print(x4_t.shape)
                    if model_stu.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        #loss = criterion(masks_pred, true_masks)+criterion_heatmap(heatmap_pred,heatmap.float())#+0.5*loss_gca
                        loss = criterion(masks_pred,true_masks,heatmap)+criterion_heatmap(heatmap_pred,heatmap)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model_stu.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model_stu.parameters(), gradient_clipping)
                #torch.nn.utils.clip_grad_norm_(model_gca.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                """
                optimizer_gca.zero_grad(set_to_none=True)
                grad_scaler_gca.scale(loss_gca).backward()
                torch.nn.utils.clip_grad_norm_(model_gca.parameters(), gradient_clipping)
                grad_scaler_gca.step(optimizer_gca)
                grad_scaler_gca.update()
                """

                #ema.update()#EMA更新教师模型
                #ema.apply_shadow()
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                """
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                """
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # Evaluation round
                division_step = (n_train // (1 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        val_score = evaluate(model_stu, model_teacher,val_loader, device, amp,'./data/imgs/')
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        """
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
                        """

        if save_checkpoint and epoch%10==0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model_stu.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_stu_epoch_{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--load_vae', '-lv', type=str, default=False, help='Load a vae model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model_teacher = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)#Teacher
    model_stu = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)#Student
    model_gca=GCALoss(img_size=112,num_head=8,patch_size=4,in_chans=512,dim=256,dropout_pro=0.0,gaze_channels=3)
    model_teacher = model_teacher.to(memory_format=torch.channels_last)
    model_stu = model_stu.to(memory_format=torch.channels_last)
    model_gca = model_gca.to(memory_format=torch.channels_last)
    
    logging.info(f'Network:\n'
                 f'\t{model_teacher.n_channels} input channels\n'
                 f'\t{model_teacher.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model_teacher.bilinear else "Transposed conv"} upscaling')
    logging.info(f'Network:\n'
                 f'\t{model_stu.n_channels} input channels\n'
                 f'\t{model_stu.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model_stu.bilinear else "Transposed conv"} upscaling')
    if args.load:
        state_dict_teacher = torch.load(args.load, map_location=device)
        state_dict_stu = torch.load(args.load, map_location=device)
        del state_dict_teacher['mask_values']
        del state_dict_stu['mask_values']
        model_stu.load_state_dict(state_dict_teacher)
        model_teacher.load_state_dict(state_dict_stu)

        logging.info(f'Model loaded from {args.load}')
        logging.info(f'Model loaded from {args.load_vae}')

    model_teacher.to(device=device)
    model_stu.to(device=device)
    model_gca.to(device=device)
    try:
        train_model(
            model_teacher=model_teacher,
            model_stu=model_stu,
            model_gca=model_gca,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model_teacher.use_checkpointing()
        model_stu.use_checkpointing()
        model_gca.use_checkpointing()
        train_model(
            model_teacher=model_teacher,
            model_stu=model_stu,
            model_gca=model_gca,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
