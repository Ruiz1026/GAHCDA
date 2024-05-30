import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw

def extract_points_from_mask(mask_path, num_points):
    # 读取mask图
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 获取非零像素点的坐标
    non_zero_points = np.transpose(np.nonzero(mask))

    # 随机采样 num_points 个点
    random_indices = np.random.choice(len(non_zero_points), num_points, replace=False)
    random_points = non_zero_points[random_indices]

    return random_points

def kmeans_sampling(points, num_samples):
    # 使用KMeans聚类获取 num_samples 个聚类中心
    kmeans = KMeans(n_clusters=num_samples, random_state=42)
    kmeans.fit(points)
    cluster_centers = kmeans.cluster_centers_

    return cluster_centers

def draw_lines_from_points(image_path, points, line_width=5):
    # 按照从下到上、再从上到下、整体从左到右的顺序排序坐标点
    points = sorted(points, key=lambda x: (x[1], x[0]))

    # 创建一个黑色图像
    image = Image.new("RGB", (224, 224), "black")
    draw = ImageDraw.Draw(image)

    # 在相邻的点之间绘制白色直线
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill="white", width=line_width)

    # 保存图像
    image.save(image_path)

# 替换这里的路径为你的mask图路径
mask_path = "your_mask_image.png"
# 设置要随机采样的点数量和最终聚类中心数量
num_random_points = 100
num_clusters = 20

# 提取坐标点
points = extract_points_from_mask(mask_path, num_random_points)

# 使用KMeans获取聚类中心
cluster_centers = kmeans_sampling(points, num_clusters)

# 保存图像的路径
image_path = "output_image.png"

# 调用函数绘制直线并保存图像，增加线宽为2
draw_lines_from_points(image_path, cluster_centers, line_width=2)
