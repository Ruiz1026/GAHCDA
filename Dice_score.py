from PIL import Image
import numpy as np
import os

def calculate_dice_coefficient(label_path, prediction_path):
    # 读取标签图和预测结果图
    label_image = Image.open(label_path)
    prediction_image = Image.open(prediction_path)

    # 转换为NumPy数组
    label_array = np.array(label_image)
    prediction_array = np.array(prediction_image)

    # 将图像转换为二进制掩模
    label_mask = (label_array > 0).astype(int)
    prediction_mask = (prediction_array > 0).astype(int)

    # 计算Dice系数
    intersection = np.sum(label_mask * prediction_mask)
    dice_coefficient = (2.0 * intersection) / (np.sum(label_mask) + np.sum(prediction_mask))

    return dice_coefficient

def calculate_average_dice_coefficient(label_folder, prediction_folder):
    # 获取文件夹中所有文件名
    file_names = os.listdir(label_folder)

    # 初始化总和和计数器
    total_dice_coefficient = 0
    count = 0

    # 计算每对标签图和预测结果图的Dice系数，并累加总和
    for file_name in file_names:
        label_path = os.path.join(label_folder, file_name)
        prediction_path = os.path.join(prediction_folder, file_name)

        dice_coefficient = calculate_dice_coefficient(label_path, prediction_path)

        total_dice_coefficient += dice_coefficient
        count += 1

    # 计算平均Dice系数
    average_dice_coefficient = total_dice_coefficient / count

    return average_dice_coefficient

# 用法示例
label_folder = "results1"
prediction_folder = "HMC-QU_gt"

average_dice = calculate_average_dice_coefficient(label_folder, prediction_folder)
print(f"Average Dice coefficient: {average_dice}")
