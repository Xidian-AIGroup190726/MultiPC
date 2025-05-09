import numpy as np
from scipy.ndimage import sobel
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

def calculate_ergas_4d(im_fusion, im_reference, l=4):
    """
    计算4维多光谱图像的ERGAS指标

    参数:
    - im_fusion: 融合后的图像 [B,H,W,C]
    - im_reference: 参考高分辨率图像 [B,H,W,C]
    - l: 低分辨率图像的比例因子（默认为8）

    返回:
    - ERGAS平均值
    """
    # 确保输入数据是4维的
    if len(im_fusion.shape) == 3:
        im_fusion = im_fusion[np.newaxis, ...]
        im_reference = im_reference[np.newaxis, ...]

    # 检查图像维度是否一致
    assert im_fusion.shape == im_reference.shape, "图像尺寸必须一致"

    batch_size, height, width, channels = im_fusion.shape
    ergas_values = []

    for b in range(batch_size):
        # 计算均方误差（MSE）
        mse_bands = []
        for i in range(channels):
            mse = np.mean((im_fusion[b, :, :, i] - im_reference[b, :, :, i]) ** 2)
            mse_bands.append(mse)

        # 计算每个波段的均值
        mean_bands = [np.mean(im_reference[b, :, :, i]) for i in range(channels)]

        # ERGAS计算
        rmse_bands = np.sqrt(mse_bands)
        relative_rmse = [rmse / mean if mean != 0 else 0 for rmse, mean in zip(rmse_bands, mean_bands)]

        ergas = 100 / l * np.sqrt(np.mean([rmse ** 2 for rmse in relative_rmse]))
        ergas_values.append(ergas)

    return np.mean(ergas_values)

def SCC_numpy(target, pred):
    """
    计算空间相关系数(SCC)
    target, pred: [H,W,C] 或 [B,H,W,C]
    """
    target = np.array(target, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)

    # 确保输入数据是4维的
    if len(target.shape) == 3:
        target = target[np.newaxis, ...]
        pred = pred[np.newaxis, ...]

    batch_size = target.shape[0]
    num_channels = target.shape[-1]
    scc_sum = 0
    valid_counts = 0  # 记录有效的计算次数

    for i in range(batch_size):
        channel_scc = 0
        valid_channels = 0  # 记录每个batch中有效的通道数

        for c in range(num_channels):
            target_channel = target[i, ..., c].flatten()
            pred_channel = pred[i, ..., c].flatten()

            # 检查是否存在有效的变化
            if np.std(target_channel) > 1e-6 and np.std(pred_channel) > 1e-6:
                try:
                    correlation, _ = pearsonr(target_channel, pred_channel)
                    if not np.isnan(correlation):  # 确保相关系数不是nan
                        channel_scc += correlation
                        valid_channels += 1
                except:
                    continue

        # 只在有有效通道时计算平均值
        if valid_channels > 0:
            scc_sum += channel_scc / valid_channels
            valid_counts += 1

    # 返回有效的平均值，如果没有有效值则返回0
    if valid_counts > 0:
        scc = scc_sum / valid_counts
        return np.clip(scc, -1, 1)
    else:
        return 0.0
def SAM_numpy(target, pred):
    """
    计算光谱角度映射器(SAM)
    target, pred: [H,W,C] 或 [B,H,W,C]
    """
    target = np.array(target, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)

    # 确保输入数据是4维的
    if len(target.shape) == 3:
        target = target[np.newaxis, ...]
        pred = pred[np.newaxis, ...]

    batch_size = target.shape[0]
    sam_sum = 0

    for i in range(batch_size):
        # 重塑为2D数组 [H*W, C]
        target_2d = target[i].reshape(-1, target.shape[-1])
        pred_2d = pred[i].reshape(-1, pred.shape[-1])

        # 计算点积
        dot_product = np.sum(target_2d * pred_2d, axis=1)
        # 计算范数
        target_norm = np.sqrt(np.sum(target_2d ** 2, axis=1))
        pred_norm = np.sqrt(np.sum(pred_2d ** 2, axis=1))

        # 避免除以零
        eps = 1e-8
        cos_angle = dot_product / (target_norm * pred_norm + eps)
        # 裁剪到[-1, 1]范围内
        cos_angle = np.clip(cos_angle, -1, 1)
        # 计算角度（弧度）并转换为度数
        angle = np.arccos(cos_angle) * 180 / np.pi
        # 计算平均角度
        sam_sum += np.mean(angle)

    return sam_sum / batch_size

# def SSIM_numpy(x_true, x_pred, data_range):
#     """
#     Args:
#         x_true (np.ndarray): target image, shape like [H, W, C]
#         x_pred (np.ndarray): predict image, shape like [H, W, C]
#         data_range (int): max_value of the image
#     Returns:
#         float: SSIM value
#     """
#     ssim = 0
#     for c in range(x_true.shape[-1]):
#         ssim += structural_similarity(x_true[:, :, c], x_pred[:, :, c], data_range=data_range)
#     return ssim / x_true.shape[-1]


def SSIM_numpy(target, pred, data_range=1):
    """
    计算多光谱图像的SSIM
    target, pred: [H,W,C] 或 [B,H,W,C]
    """
    target = np.array(target, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)

    # 确保输入数据是4维的
    if len(target.shape) == 3:
        target = target[np.newaxis, ...]
        pred = pred[np.newaxis, ...]

    batch_size = target.shape[0]
    ssim_sum = 0

    for i in range(batch_size):
        ssim = structural_similarity(target[i], pred[i], data_range=data_range, channel_axis=-1,win_size=3)
        ssim_sum += ssim

    return ssim_sum / batch_size


# def MPSNR_numpy(x_true, x_pred, data_range=1):
#     """
#     Args:
#         x_true (np.ndarray): target image, shape like [H, W, C]
#         x_pred (np.ndarray): predict image, shape like [H, W, C]
#         data_range (int): max_value of the image
#     Returns:
#         float: Mean PSNR value
#     """
#     tmp = []
#     for c in range(x_true.shape[-1]):
#         mse = np.mean((x_true[:, :, c] - x_pred[:, :, c]) ** 2)
#         psnr = 10 * np.log10(data_range ** 2 / mse)
#         tmp.append(psnr)
#     return np.mean(tmp)


def MPSNR_numpy(target, pred, data_range=1):
    """
    计算多光谱图像的PSNR
    target, pred: [H,W,C] 或 [B,H,W,C]
    """
    target = np.array(target, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)

    # 确保输入数据是4维的
    if len(target.shape) == 3:
        target = target[np.newaxis, ...]
        pred = pred[np.newaxis, ...]

    batch_size = target.shape[0]
    psnr_sum = 0

    for i in range(batch_size):
        mse = np.mean((target[i] - pred[i]) ** 2)
        if mse == 0:
            psnr_sum += 100
        else:
            psnr_sum += 20 * np.log10(data_range / np.sqrt(mse))

    return psnr_sum / batch_size