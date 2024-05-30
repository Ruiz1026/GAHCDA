import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.data_loading import BasicDataset
import logging
from PIL import Image
import numpy as np


@torch.inference_mode()
def get_output_filenames(input_files):
    def _generate_name(fn):
        return f'temp_results/{os.path.splitext(fn)[0]}.png'

    return list(map(_generate_name, os.listdir(input_files)))
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output,x1,x2,x3,x4,x5 = net(img)
        output = output.cpu().to(torch.float32)
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()
def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)
def convert_and_save_gif(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 打开图像
        img = Image.open(input_path)

        # 将图像转换为单通道灰度图像
        img_gray = img.convert("L")

        # 将图像转换为NumPy数组
        img_array = np.array(img_gray)

        # 将元素值为255的元素转换为1
        img_array[img_array == 255] = 1

        # 将NumPy数组转换回图像
        result_img = Image.fromarray(img_array)

        # 构建输出路径
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.gif")

        # 保存单独的GIF文件
        result_img.save(output_path, save_all=True, duration=500)
def evaluate(net,net_teacher,dataloader, device, amp,input_files):
    net.eval()
    net_teacher.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            images, mask_true,heatmap = batch['image'], batch['mask'],batch['heatmap']
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            heatmap = heatmap.to(device=device, dtype=torch.long)
            # predict the mask
            mask_pred,heatmap_pred,x1,x2,x3,x4,x5 = net(image)
            heatmap_pred = heatmap_pred.to(device=device, dtype=torch.long)
            # 将张量转换为 PIL 图像
            heat_map_image = transforms.ToPILImage()(heatmap_pred)
            # 保存图像到本地
            heat_map_image.save('temp_heatmap/heat_map_image.png')
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
        if epoch>=1000:
            in_files = os.listdir(input_files)
            out_files = get_output_filenames(input_files)
            for i, filename in enumerate(in_files):
                img = Image.open(input_files+'/'+filename)

                mask = predict_img(net=net_teacher,
                                full_img=img,
                                scale_factor=0.5,
                                out_threshold=0.5,
                                device=device)
                out_filename = out_files[i]
                mask_values=[0,1]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
            convert_and_save_gif('temp_results/', './data/pseudo_label/')
    net.train()
    net_teacher.train()
    return dice_score / max(num_val_batches, 1)


        

