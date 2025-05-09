# main.py
# ================== Imports ==================
import os
import copy
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tifffile import imread
from tqdm import tqdm
from data_utils import (
    load_images, normalize_image, pad_images, load_labels,
    prepare_label_indices, split_indices, convert_images_to_tensor
)
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
# Custom modules
from method import MPSNR_numpy, SAM_numpy, SCC_numpy, SSIM_numpy, calculate_ergas_4d
from model import MultiTaskModel

# ================== Configuration ==================
os.environ["OMP_NUM_THREADS"] = "1"
device = torch.device('cuda')
writer = SummaryWriter(log_dir='runs/train')

# Dataset paths
DATASET_NAME = "hhht"
PAN_PATH = f"./{DATASET_NAME}/data/pan.tif"
MS4_PATH = f"./{DATASET_NAME}/data/ms4.tif"
TRAIN_LABEL_PATH = f"./{DATASET_NAME}/data/train.npy"
TEST_LABEL_PATH = f"./{DATASET_NAME}/data/test.npy"

# Training parameters
TRAIN_RATE = 1.0
VAL_RATE = 0.10
BATCH_SIZE = 128
EPOCHS = 1
WAVELET = 'haar'
INIT_LR = 1e-4
PATCH_SIZE = 16

# Loss weights
WEIGHT_CLS = 1.0
WEIGHT_REC = 1.0
WEIGHT_L1_HP = 0.01
WEIGHT_CONTRAST = 0.05

BEGIN = 0.01
K = 0.001

# ================== Utility Functions ==================
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# ================== Dataset Preparation ==================
# The image and label loading/preprocessing
# ... [Omitted here for brevity, should be moved to a separate module for clarity]

# ================== Dataset Class Definitions ==================
class MyData(Dataset):
    def __init__(self, ms4, pan, labels, coords, cut_size):
        self.ms4 = ms4
        self.pan = pan
        self.labels = labels
        self.coords = coords
        self.cut_ms = cut_size
        self.cut_pan = cut_size * 4

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x_ms, y_ms = self.coords[idx]
        x_pan, y_pan = 4 * x_ms, 4 * y_ms
        ms_patch = self.ms4[:, x_ms:x_ms + self.cut_ms, y_ms:y_ms + self.cut_ms]
        pan_patch = self.pan[:, x_pan:x_pan + self.cut_pan, y_pan:y_pan + self.cut_pan]
        return ms_patch, pan_patch, self.labels[idx], self.coords[idx]

class MyDataTest(Dataset):
    def __init__(self, ms4, pan, coords, cut_size):
        self.ms4 = ms4
        self.pan = pan
        self.coords = coords
        self.cut_ms = cut_size
        self.cut_pan = cut_size * 4

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x_ms, y_ms = self.coords[idx]
        x_pan, y_pan = 4 * x_ms, 4 * y_ms
        ms_patch = self.ms4[:, x_ms:x_ms + self.cut_ms, y_ms:y_ms + self.cut_ms]
        pan_patch = self.pan[:, x_pan:x_pan + self.cut_pan, y_pan:y_pan + self.cut_pan]
        return ms_patch, pan_patch, self.coords[idx]

# ================== Custom Loss ==================
class ContrastiveLoss(nn.Module):
    def __init__(self, input_dim=16384, feat_dim=256):
        super().__init__()
        self.proj_head_h = nn.Linear(input_dim, feat_dim)
        self.proj_head_l = nn.Linear(input_dim, feat_dim)

    def forward(self, hc, lc, temperature=0.1):
        if hc.dim() == 4:
            hc, lc = hc.view(hc.size(0), -1), lc.view(lc.size(0), -1)
        hc_proj = F.normalize(self.proj_head_h(hc), p=2, dim=1)
        lc_proj = F.normalize(self.proj_head_l(lc), p=2, dim=1)
        pos_sim = torch.sum(hc_proj * lc_proj, dim=1) / temperature
        neg_sim = torch.mm(hc_proj, torch.cat([hc_proj, lc_proj], dim=0).T) / temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(hc.size(0), dtype=torch.long, device=hc.device)
        return F.cross_entropy(logits, labels)

# ================== Training & Evaluation ==================
def train(model, dataloader, device, optimizer, epoch, weightp, begin, k, model_old):
    model.train()
    correct, total = 0.0, 0
    a, b = begin, begin + epoch * k
    contrastive_loss = ContrastiveLoss().to(device)

    for step, (ms, pan, label, _) in enumerate(tqdm(dataloader, desc=f"Train Epoch {epoch}")):
        ms, pan, label = ms.to(device), pan.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(ms, pan, device)
        outc, outp = outputs['outc'], outputs['outp']
        hc2, hp2, l2 = outputs['features']['high']['hc2'], outputs['features']['pan']['hp2'], outputs['features']['low']['l2']

        loss_cls = F.cross_entropy(outc, label)
        loss_rec = F.l1_loss(outp, ms)
        loss_hp = F.l1_loss(hp2, ms)
        loss_contrast = contrastive_loss(hc2, l2)

        loss_distill = torch.tensor(0.0, device=device)
        if epoch > 1:
            with torch.no_grad():
                old_outputs = model_old(ms, pan, device)
                mask = old_outputs['outc'].max(1)[1] == label
            if mask.any():
                new_probs = F.log_softmax(outc[mask], dim=1)
                old_probs = F.softmax(old_outputs['outc'][mask], dim=1)
                loss_distill = a * F.kl_div(new_probs, old_probs, reduction='batchmean') + b * F.l1_loss(outp, old_outputs['outp']) / BATCH_SIZE

        total_loss = (
            WEIGHT_CLS * loss_cls + weightp * loss_rec + loss_distill + WEIGHT_L1_HP * loss_hp + WEIGHT_CONTRAST * loss_contrast
        )

        total_loss.backward()
        optimizer.step()
        total += label.size(0)
        correct += outc.max(1)[1].eq(label).sum().item()

    print(f"Train Accuracy: {100.0 * correct / total:.2f}%")



def kappa(conf_matrix):
    n = np.sum(conf_matrix)
    sum_po = np.trace(conf_matrix)
    sum_pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1))
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)

def test(model, dataloader, device, weightp, num_classes):
    model.eval()
    correct = 0.0
    total = 0

    all_preds = []
    all_labels = []

    psnr_values, ssim_values, sam_values, scc_values, ergas_values = [], [], [], [], []
    contrastive_loss = ContrastiveLoss().to(device)

    with torch.no_grad():
        for ms, pan, label, _ in tqdm(dataloader, desc="Testing", colour='green'):
            ms, pan, label = ms.to(device), pan.to(device), label.to(device)
            outputs = model(ms, pan, device)

            outc, outp = outputs['outc'], outputs['outp']

            total += label.size(0)
            pred = outc.argmax(1)
            correct += pred.eq(label).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            # 图像质量指标
            hrms = ms.cpu().numpy().transpose(0, 2, 3, 1)[0]
            sample = outp.cpu().numpy().transpose(0, 2, 3, 1)[0]

            psnr_values.append(MPSNR_numpy(hrms, sample, data_range=1))
            ssim_values.append(SSIM_numpy(hrms, sample, data_range=255))
            sam_values.append(SAM_numpy(hrms, sample))
            scc_values.append(SCC_numpy(hrms, sample))
            ergas_values.append(calculate_ergas_4d(sample[np.newaxis], hrms[np.newaxis]))

    # 统计分类性能
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    per_class_acc = conf_matrix.diagonal() / np.maximum(conf_matrix.sum(axis=1), 1)
    average_accuracy = np.mean(per_class_acc)
    overall_accuracy = np.sum(conf_matrix.diagonal()) / np.sum(conf_matrix)
    kappa_score = kappa(conf_matrix)

    # 输出分类结果
    print(f'weightp: {weightp}')
    print(f'AA: {average_accuracy:.4f}')
    print(f'OA: {overall_accuracy:.4f}')
    print(f'Kappa Coefficient: {kappa_score:.4f}')

    # 输出图像质量指标
    print(f'Average PSNR: {np.mean(psnr_values):.4f}')
    print(f'Average SSIM: {np.mean(ssim_values):.4f}')
    print(f'Average SAM: {np.mean(sam_values):.4f}')
    print(f'Average SCC: {np.mean(scc_values):.4f}')
    print(f'Average ERGAS: {np.mean(ergas_values):.4f}')

    return 100.0 * correct / total

# ================== Main ==================
if __name__ == '__main__':
    # Load and preprocess data
    pan_np, ms4_np = load_images(PAN_PATH, MS4_PATH)
    ms4_np, pan_np = pad_images(ms4_np, pan_np, PATCH_SIZE)
    ms4_np = normalize_image(ms4_np)
    pan_np = normalize_image(pan_np)
    label_train_np, label_test_np = load_labels(TRAIN_LABEL_PATH, TEST_LABEL_PATH)

    coords_train, num_classes, _, _ = prepare_label_indices(label_train_np)
    coords_test, _, _, _ = prepare_label_indices(label_test_np)

    train_xy, val_xy, test_xy, label_train, label_val, label_test = split_indices(coords_train, TRAIN_RATE, VAL_RATE)
    ms4_tensor, pan_tensor = convert_images_to_tensor(ms4_np, pan_np)

    # Create datasets and dataloaders
    train_data = MyData(ms4_tensor, pan_tensor, label_train, train_xy, PATCH_SIZE)
    val_data = MyData(ms4_tensor, pan_tensor, label_val, val_xy, PATCH_SIZE)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model and optimizer
    model = MultiTaskModel(num_classes=num_classes, Wavelet=WAVELET).to(device)
    model_old = copy.deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, device, optimizer, epoch, WEIGHT_REC, BEGIN, K, model_old)
        if epoch == EPOCHS:
            test(model, val_loader, device, weightp=WEIGHT_REC, num_classes=num_classes)
            torch.save(model.state_dict(), f'final_model_{DATASET_NAME}.pth')
        model_old = copy.deepcopy(model)
