import os
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def compute_asd(prediction, ground_truth):
    # Find boundary pixels in prediction and ground truth
    prediction_boundary = find_boundary(prediction)
    ground_truth_boundary = find_boundary(ground_truth)
    
    if len(prediction_boundary) == 0 or len(ground_truth_boundary) == 0:
        return 0
    
    # Compute distances
    distances_pred_to_gt = directed_hausdorff(prediction_boundary, ground_truth_boundary)[0]
    distances_gt_to_pred = directed_hausdorff(ground_truth_boundary, prediction_boundary)[0]
    
    # Compute ASD
    asd = (distances_pred_to_gt + distances_gt_to_pred) / 2.0
    
    return asd


def find_boundary(image):
    boundary = []
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            if image[i, j] == 255:
                if (i == 0 or i == h - 1 or j == 0 or j == w - 1 or
                    image[i - 1, j] == 0 or image[i + 1, j] == 0 or
                    image[i, j - 1] == 0 or image[i, j + 1] == 0):
                    boundary.append([i, j])
    return np.array(boundary)

def read_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Function to compute ASD for all pairs of prediction and ground truth images
def compute_average_asd_for_folder(pred_folder, gt_folder):
    prediction_images = read_images_from_folder(pred_folder)
    ground_truth_images = read_images_from_folder(gt_folder)
    
    total_asd = 0.0
    num_pairs = min(len(prediction_images), len(ground_truth_images))
    
    for pred_img, gt_img in zip(prediction_images, ground_truth_images):
        asd = compute_asd(pred_img, gt_img)
        print(asd)
        total_asd += asd
    
    average_asd = total_asd / num_pairs
    return average_asd

# Example usage
pred_folder = "assd/supervised"
gt_folder = "HMC-QU_gt"

average_asd = compute_average_asd_for_folder(pred_folder, gt_folder)
print("Average ASSD:", average_asd)
