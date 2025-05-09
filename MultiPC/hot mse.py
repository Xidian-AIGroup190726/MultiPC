import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import os

def read_tif_band(path, band_index):
    dataset = gdal.Open(path)
    if dataset is None:
        raise FileNotFoundError(f"无法打开文件: {path}")
    band = dataset.GetRasterBand(band_index + 1)  # gdal从1开始计数
    arr = band.ReadAsArray().astype(np.float32)
    return arr
def compute_and_plot_rmse_gdal(gt_path, pred_path, output_path, channel=0, vmin=None, vmax=None):
    gt_band = read_tif_band(gt_path, channel)
    pred_band = read_tif_band(pred_path, channel)

    print(f"📊 GT 范围: min={gt_band.min():.2f}, max={gt_band.max():.2f}")
    print(f"📊 Pred 范围: min={pred_band.min():.2f}, max={pred_band.max():.2f}")

    rmse_map = np.sqrt((gt_band - pred_band) ** 2)
    print(f"📈 RMSE 范围: min={rmse_map.min():.2f}, max={rmse_map.max():.2f}")

    # 如果没提供，就自己计算
    if vmin is None:
        vmin = np.percentile(rmse_map, 1)
    if vmax is None:
        vmax = np.percentile(rmse_map, 99)

    print(f"🎯 Colorbar 范围: vmin={vmin:.2f}, vmax={vmax:.2f}")

    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(rmse_map, cmap='YlGnBu', vmin=vmin, vmax=vmax)
    plt.axis('off')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    filename = f"{os.path.basename(gt_path).split('.')[0]}_ch{channel}_rmse.png"
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"✅ 已保存 RMSE 热力图: {os.path.join(output_path, filename)}\n")

datasetname = 'sh'
gt_path = "./output/sh_512gt.tif"
pred_path = "./aa/sh_512.tif"
output_path = "./aa"
# pred_path = "./output/sh_512.tif"
# output_path = "./output_heatmaps"
compute_and_plot_rmse_gdal(gt_path, pred_path, output_path,channel=2)
