# data_utils.py
import numpy as np
import cv2
from tifffile import imread
import torch


def load_images(pan_path, ms4_path):
    pan_np = imread(pan_path)
    ms4_np = imread(ms4_path)
    return pan_np, ms4_np


def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def pad_images(ms4_np, pan_np, ms_patch_size=16, border_type=cv2.BORDER_REFLECT_101):
    top_ms = int(ms_patch_size / 2 - 1)
    bottom_ms = int(ms_patch_size / 2)
    left_ms = top_ms
    right_ms = bottom_ms
    ms4_padded = cv2.copyMakeBorder(ms4_np, top_ms, bottom_ms, left_ms, right_ms, border_type)

    pan_patch_size = ms_patch_size * 4
    top_pan = int(pan_patch_size / 2 - 4)
    bottom_pan = int(pan_patch_size / 2)
    left_pan = top_pan
    right_pan = bottom_pan
    pan_padded = cv2.copyMakeBorder(pan_np, top_pan, bottom_pan, left_pan, right_pan, border_type)

    return ms4_padded, pan_padded


def load_labels(train_label_path, test_label_path):
    label_train = np.load(train_label_path).astype(np.uint8) - 1
    label_test = np.load(test_label_path).astype(np.uint8) - 1
    return label_train, label_test


def prepare_label_indices(label_np):
    label_row, label_col = label_np.shape
    categories = len(np.unique(label_np[label_np != 255]))
    label_coords = [[] for _ in range(categories)]

    for i in range(label_row):
        for j in range(label_col):
            label = label_np[i, j]
            if label != 255:
                label_coords[label].append([i, j])

    for i in range(categories):
        coords = np.array(label_coords[i])
        indices = np.arange(len(coords))
        np.random.shuffle(indices)
        label_coords[i] = coords[indices]

    return label_coords, categories, label_row, label_col


def split_indices(label_coords, train_rate=1.0, val_rate=0.1):
    train_coords, val_coords, test_coords = [], [], []
    label_train, label_val, label_test = [], [], []

    for cat, coords in enumerate(label_coords):
        n_total = len(coords)
        n_train = int(n_total * train_rate)
        n_val = int(n_total * val_rate)

        train_coords.extend(coords[:n_train])
        val_coords.extend(coords[:n_val])
        test_coords.extend(coords)

        label_train.extend([cat] * n_train)
        label_val.extend([cat] * n_val)
        label_test.extend([cat] * n_total)

    def to_tensor(arr):
        return torch.from_numpy(np.array(arr)).long()

    train_coords, val_coords, test_coords = map(to_tensor, [train_coords, val_coords, test_coords])
    label_train, label_val, label_test = map(to_tensor, [label_train, label_val, label_test])

    # 对合并后的训练/验证/测试进行整体乱序
    def shuffle(x, y):
        idx = torch.randperm(len(x))
        return x[idx], y[idx]

    train_coords, label_train = shuffle(train_coords, label_train)
    val_coords, label_val = shuffle(val_coords, label_val)
    test_coords, label_test = shuffle(test_coords, label_test)

    return train_coords, val_coords, test_coords, label_train, label_val, label_test


def convert_images_to_tensor(ms4_np, pan_np):
    ms4_tensor = torch.from_numpy(ms4_np.transpose(2, 0, 1)).float()
    pan_tensor = torch.from_numpy(np.expand_dims(pan_np, axis=0)).float()
    return ms4_tensor, pan_tensor