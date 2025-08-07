import os
import random
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from torchmetrics import F1Score
from PIL import Image

# ====== Установка random seed ======
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ====== ПАРАМЕТРЫ ======
DATA_DIR = 'data/train'
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
NUM_FOLDS = 5
SEED_BASE = 42
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLES_PER_IMG = 20
VAL_SAMPLES_PER_IMG = 20
IMG_EXTS = ('*.png','*.jpg','*.jpeg')

# ====== BiasedRandomResizedCrop ======
class BiasedRandomResizedCrop:
    def __init__(self, size, scale=(0.3,1.0), ratio=(3/4,4/3), alpha=5, beta=2):
        self.size = (size, size) if isinstance(size, int) else size
        self.scale, self.ratio = scale, ratio
        self.alpha, self.beta = alpha, beta
    def get_params(self, img):
        W, H = img.size; area = W*H
        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            ar = random.uniform(*self.ratio)
            w = int(round((target_area * ar)**0.5)); h = int(round((target_area / ar)**0.5))
            if w <= W and h <= H:
                x = random.randint(0, W-w)
                yc = np.random.beta(self.alpha, self.beta)
                y = int((H-h) * yc)
                return x, y, w, h
        w = min(W,H); x = (W-w)//2; y = (H-w)//2; return x, y, w, w
    def __call__(self, img):
        x, y, w, h = self.get_params(img)
        return transforms.functional.resized_crop(img, y, x, h, w, self.size)

# ====== Датасет для списка путей ======
class IllumListDataset(Dataset):
    def __init__(self, img_paths, labels, samples_per_img, transform):
        self.img_paths = img_paths; self.labels = labels
        self.samples_per_img = samples_per_img
        self.transform = transform
    def __len__(self): return len(self.img_paths) * self.samples_per_img
    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_img
        img = Image.open(self.img_paths[img_idx]).convert('RGB')
        x = self.transform(img)
        y = self.labels[img_idx]
        return x, torch.tensor(y, dtype=torch.float32)

# ====== Модель ======
def get_model():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_f, 1)
    return model

# ====== Сбор файлов ======
def prepare_file_lists():
    paths, labels = [], []
    for lbl_name, lbl_val in [('night',0), ('day',1)]:
        for ext in IMG_EXTS:
            for p in glob(f"{DATA_DIR}/{lbl_name}/{ext}"):
                paths.append(p); labels.append(lbl_val)
    return np.array(paths), np.array(labels)

# ====== Обучение ансамбля ======
def train_ensemble():
    paths, labels = prepare_file_lists()
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED_BASE)

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels), start=1):
        print(f"\n=== Fold {fold}/{NUM_FOLDS} ===")
        set_seed(SEED_BASE + fold)

        train_paths, train_labels = paths[train_idx], labels[train_idx]
        val_paths, val_labels     = paths[val_idx], labels[val_idx]

        train_tf = transforms.Compose([
            BiasedRandomResizedCrop(224), transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
        ])
        val_tf = transforms.Compose([
            BiasedRandomResizedCrop(224), transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
        ])

        train_ds = IllumListDataset(train_paths, train_labels, SAMPLES_PER_IMG, train_tf)
        val_ds   = IllumListDataset(val_paths,   val_labels,   VAL_SAMPLES_PER_IMG, val_tf)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4)

        model = get_model().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        pos_weight = torch.tensor([(len(train_labels) - train_labels.sum()) / train_labels.sum()]).to(DEVICE)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        f1_metric = F1Score(task="binary", threshold=0.5).to(DEVICE)

        best_f1, best_loss, best_state = 0.0, float('inf'), None

        for epoch in range(1, EPOCHS+1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval(); f1_metric.reset(); val_losses, all_preds, all_targets = [], [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits = model(x).squeeze(1)
                    val_losses.append(criterion(logits, y).item())
                    preds = torch.sigmoid(logits)
                    f1_metric.update(preds, y.int())
            val_f1 = f1_metric.compute().item()
            val_loss = float(np.mean(val_losses))
            print(f"Fold {fold} Ep{epoch:02d}: Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")

            if (val_f1 > best_f1) or (val_f1 == best_f1 and val_loss < best_loss):
                best_f1, best_loss = val_f1, val_loss
                best_state = model.state_dict()

        ckpt_name = f"mobilenetv3_fold{fold}_f1{best_f1:.4f}_loss{best_loss:.4f}.pt"
        torch.save(best_state, os.path.join(CHECKPOINT_DIR, ckpt_name))
        print(f"Saved best model for fold {fold}: {ckpt_name}")

    print("\nAll folds completed.")

if __name__ == '__main__':
    train_ensemble()