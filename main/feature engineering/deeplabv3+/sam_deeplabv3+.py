import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import sys
import os
from PIL import Image
from collections import Counter
from sklearn.metrics import confusion_matrix
import multiprocessing as mp
from tqdm import tqdm
sys.path.append("..")


# Add legends on the Deeplabv3+ prediction results
def image_annotation(file_path):
    # define the label and color
    # 19 classes
    cityscapes_labels = ['road',
                         'sidewalk',
                         'building',
                         'wall',
                         'fence',
                         'pole',
                         'traffic light',
                         'traffic sign',
                         'vegetation',
                         'terrain',
                         'sky',
                         'person',
                         'rider',
                         'car',
                         'truck',
                         'bus',
                         'train',
                         'motorcycle',
                         'bicycle']

    # color of corresponding labels
    cityscapes_colors = np.array([[128, 64, 128],
                                  [244, 35, 232],
                                  [70, 70, 70],
                                  [102, 102, 156],
                                  [190, 153, 153],
                                  [153, 153, 153],
                                  [250, 170, 30],
                                  [220, 220, 0],
                                  [107, 142, 35],
                                  [152, 251, 152],
                                  [70, 130, 180],
                                  [220, 20, 60],
                                  [255, 0, 0],
                                  [0, 0, 142],
                                  [0, 0, 70],
                                  [0, 60, 100],
                                  [0, 80, 100],
                                  [0, 0, 230],
                                  [119, 11, 32]])

    # cityscapes_colors = np.array(cityscapes_colors) / 255.0

    # 加载预测结果图像和标签
    prediction = Image.open(file_path)
    image_array = np.asarray(prediction)

    # 图片里的RGB颜色
    unique_colors = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0)

    # 返回颜色列表里包含图片RGB颜色的索引
    indices = [np.where(np.all(cityscapes_colors == unique_color, axis=1))[0][0] for unique_color in unique_colors]
    img_label = [cityscapes_labels[i] for i in indices]

    # 绘制预测结果图像和标签
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.imshow(prediction)

    unique_colors = np.array(unique_colors) / 255.0

    patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=unique_colors[i],
                        label="{:s}".format(img_label[i]))[0] for i in range(len(img_label))]
    plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper right', fontsize=8)
    plt.show()


# Use of SAM masks for voting-based refinement
# types parameter: gsv/panorama/validation
def sam_semantic_segmentation(name, mask_threshold, types):

    try:
        # Segment image by Deeplabv3+
        seg_result = cv.imread('./{}/{}_results/{}'.format(types, types, name))

        # Read the mask info
        mask_info = pd.read_csv('./{}/{}_masks/{}/metadata.csv'.format(types, types, name.split('.')[0]))

        for i in range(len(mask_info)):
            mask = cv.imread('./{}/{}_masks/{}/{}.png'.format(types, types, name.split('.')[0], i))

            # 阈值化黑白图片
            thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)[1]

            # 获取白色区域（mask)
            mask_region = np.where(thresh == 255)

            seg_info = []

            for j in range(len(mask_region[0])):
                x = mask_region[0][j]
                y = mask_region[1][j]
                b, g, r = seg_result[x, y]
                seg_info.append((r, g, b))

            color_dict = {}
            count = Counter(seg_info)
            for item, cnt in count.items():
                color_dict[item] = cnt

            max_item = max(color_dict.items(), key=lambda x: x[1])[0]
            r = max_item[0]
            g = max_item[1]
            b = max_item[2]

            ratio = color_dict[max_item] / sum(color_dict.values())

            # Transform the pixel type if larger than the threshold
            if ratio >= mask_threshold:  # Yes, transform
                for j in range(len(mask_region[0])):
                    x = mask_region[0][j]
                    y = mask_region[1][j]
                    seg_result[x, y] = [b, g, r]
            else:  # No, not change
                pass
        cv.imwrite('./{}/sam_{}/{}'.format(types, types, name), seg_result)
    except Exception:
        pass


# Get green view index, building view index, and sky view index
def pixel_index(img_path, img):
    sky = [70, 130, 180]
    building = [70, 70, 70]
    vegetation = [107, 142, 35]

    img = cv.imread(img_path + img)
    height = img.shape[0]
    width = img.shape[1]
    max_pixels = height * width
    sky_count = 0
    building_count = 0
    vegetation_count = 0

    for i in range(height):
        for j in range(width):
            if img[i, j][0] == sky[2] and img[i, j][1] == sky[1] and img[i, j][2] == sky[0]:
                sky_count += 1
            elif img[i, j][0] == building[2] and img[i, j][1] == building[1] and img[i, j][2] == building[0]:
                building_count += 1
            elif img[i, j][0] == vegetation[2] and img[i, j][1] == vegetation[1] and img[i, j][2] == vegetation[0]:
                vegetation_count += 1

    svi = "%.3f" % (sky_count / max_pixels)
    bvi = "%.3f" % (building_count / max_pixels)
    gvi = "%.3f" % (vegetation_count / max_pixels)
    return svi, bvi, gvi


# Whether SAM improve the segmentation results
def sam_performance(pred_img, true_img, building=[70, 70, 70], sky=[70, 130, 180], vegetation=[107, 142, 35]):

    pred = cv.imread(pred_img, cv.IMREAD_COLOR)
    true = cv.imread(true_img)

    pred_class = np.empty([pred.shape[0], pred.shape[1]])
    true_class = np.empty([true.shape[0], true.shape[1]])

    height = pred.shape[0]
    width = pred.shape[1]

    for i in range(height):
        for j in range(width):
            # building
            if pred[i, j][0] == building[2] and pred[i, j][1] == building[1] and pred[i, j][2] == building[0]:
                pred_class[i, j] = 1

            # sky
            elif pred[i, j][0] == sky[2] and pred[i, j][1] == sky[1] and pred[i, j][2] == sky[0]:
                pred_class[i, j] = 2

            # vegetation
            elif pred[i, j][0] == vegetation[2] and pred[i, j][1] == vegetation[1] and pred[i, j][2] == vegetation[0]:
                pred_class[i, j] = 3

            # building
            if true[i, j][0] == building[2] and true[i, j][1] == building[1] and true[i, j][2] == building[0]:
                true_class[i, j] = 1

            # sky
            elif true[i, j][0] == sky[2] and true[i, j][1] == sky[1] and true[i, j][2] == sky[0]:
                true_class[i, j] = 2

            # vegetation
            elif true[i, j][0] == vegetation[2] and true[i, j][1] == vegetation[1] and true[i, j][2] == vegetation[0]:
                true_class[i, j] = 3

    # 计算混淆矩阵
    true = true_class.flatten()
    pred = pred_class.flatten()
    cm = confusion_matrix(true, pred)

    iou_list = []
    # 提取指定类别的行和列
    for k in [1, 2, 3]:
        try:
            tp = np.diag(cm)[k]
            fp = np.sum(cm[:, k]) - tp
            fn = np.sum(cm[k, :]) - tp

            # 计算IoU（交并比）
            iou = tp / (tp + fp + fn)
        except Exception:
            iou = 0
        iou_list.append(iou)
    miou = np.array(iou_list).sum() / np.count_nonzero(iou_list)

    # building, sky, vegetation
    return iou_list[0], iou_list[1], iou_list[2], miou
