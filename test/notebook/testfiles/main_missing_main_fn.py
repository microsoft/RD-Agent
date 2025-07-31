import os
import sys
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Run in debug mode')
args = parser.parse_args()
DEBUG = args.debug

SEED = 2024
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = './workspace_input/train/'
TEST_DIR = './workspace_input/test/'
TRAIN_CSV = './workspace_input/train.csv'
SAMPLE_SUB_PATH = './workspace_input/sample_submission.csv'
MODEL_DIR = 'models/'
os.makedirs(MODEL_DIR, exist_ok=True)

def print_eda(train_df):
    print("=== Start of EDA part ===")
    print("Shape of train.csv:", train_df.shape)
    print("First 5 rows:\n", train_df.head())
    print("Column data types:\n", train_df.dtypes)
    print("Missing values per column:\n", train_df.isnull().sum())
    print("Unique values per column:")
    for col in train_df.columns:
        print(f" - {col}: {train_df[col].nunique()}")
    label_counts = train_df['has_cactus'].value_counts()
    print("Label distribution (has_cactus):")
    print(label_counts)
    pos, neg = label_counts.get(1, 0), label_counts.get(0, 0)
    total = pos + neg
    if total > 0:
        print(f"  Positive:Negative ratio: {pos}:{neg} ({pos/total:.3f}:{neg/total:.3f})")
        print(f"  Percentage positive: {pos/total*100:.2f}%")
    else:
        print("  No data found.")
    print("Image filename examples:", train_df['id'].unique()[:5])
    print("=== End of EDA part ===")

class CactusDataset(Dataset):
    def __init__(self, image_ids, labels=None, id2path=None, transforms=None):
        self.image_ids = image_ids
        self.labels = labels
        self.id2path = id2path
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.id2path[img_id]
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Cannot read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        if self.labels is not None:
            label = self.labels[idx]
            return image, label, img_id
        else:
            return image, img_id

def get_transforms(mode='train'):
    # Correct Cutout: Albumentations v1.4.15 provides 'Cutout' as a class, but not always in the root.
    # Defensive import; fallback to the most robust method for v1.4.15
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if mode == 'train':
        min_frac, max_frac = 0.05, 0.2
        min_cut = int(300 * min_frac)
        max_cut = int(300 * max_frac)
        # There is no A.Cutout in v1.4.15 root, but A.augmentations.transforms.Cutout exists.
        try:
            from albumentations.augmentations.transforms import Cutout
            have_cutout = True
        except ImportError:
            have_cutout = False
        this_cut_h = random.randint(min_cut, max_cut)
        this_cut_w = random.randint(min_cut, max_cut)
        cutout_fill = [int(255 * m) for m in imagenet_mean]
        tforms = [
            A.RandomResizedCrop(300, 300, scale=(0.7, 1.0), ratio=(0.8, 1.2), p=1.0),
            A.Rotate(limit=30, p=0.8),
        ]
        if have_cutout:
            tforms.append(
                Cutout(
                    num_holes=1,
                    max_h_size=this_cut_h,
                    max_w_size=this_cut_w,
                    fill_value=cutout_fill,  # RGB image in albumentations requires [R,G,B]
                    always_apply=False,
                    p=0.7
                )
            )
        else:
            # No available Cutout, so fallback to no cutout but emit warning
            print("WARNING: albumentations.Cutout not found, continuing without Cutout augmentation")
        tforms.extend([
            A.RandomContrast(limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.1),
            A.Normalize(mean=imagenet_mean, std=imagenet_std, max_pixel_value=255.0),
            ToTensorV2()
        ])
        return A.Compose(tforms)
    else:
        return A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=imagenet_mean, std=imagenet_std, max_pixel_value=255.0),
            ToTensorV2()
        ])

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def get_efficientnet_b3(dropout_rate=0.3):
    model = timm.create_model('efficientnet_b3', pretrained=True)
    n_in = model.classifier.in_features if hasattr(model, "classifier") else model.fc.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(n_in, 1)
    )
    return model

def compute_class_weight(y):
    counts = np.bincount(y)
    if len(counts) < 2:
        counts = np.pad(counts, (0, 2-len(counts)), constant_values=0)
    n_pos, n_neg = counts[1], counts[0]
    total = n_pos + n_neg
    minority, majority = min(n_pos, n_neg), max(n_pos, n_neg)
    ratio = majority / (minority + 1e-10)
    need_weights = ratio > 2
    weights = None
    if need_weights:
        inv_freq = [1 / (n_neg + 1e-10), 1 / (n_pos + 1e-10)]
        s = sum(inv_freq)
        weights = [w / s * 2 for w in inv_freq]
    return weights, n_pos, n_neg, ratio, need_weights

def train_one_epoch(model, loss_fn, optimizer, scheduler, dataloader, device, class_weights):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        logits = model(images)
        if class_weights is not None:
            weight = labels * class_weights[1] + (1 - labels) * class_weights[0]
            loss = loss_fn(logits, labels)
            loss = (loss * weight).mean()
        else:
            loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    return avg_loss

@torch.no_grad()
def eval_model(model, loss_fn, dataloader, device, class_weights):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        y_true.append(labels.cpu().numpy())
        y_pred.append(probs.cpu().numpy())
        if class_weights is not None:
            weight = labels * class_weights[1] + (1 - labels) * class_weights[0]
            loss = loss_fn(logits, labels)
            loss = (loss * weight).mean()
        else:
            loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    y_true = np.vstack(y_true).reshape(-1)
    y_pred = np.vstack(y_pred).reshape(-1)
    avg_loss = total_loss / total_samples
    return avg_loss, y_true, y_pred

def confusion_info(y_true, y_pred, threshold=0.5):
    preds = (y_pred > threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    return cm

def inference_and_submission(train_df, train_id2path, test_img_ids, test_id2path, dropout_rate, class_weights, need_weights,
                            BATCH_SIZE, N_WORKERS, cv_fold):
    oof_true, oof_pred, fold_scores, fold_val_ids = [], [], [], []
    for fold in range(cv_fold):
        df_val = train_df[train_df['fold'] == fold].reset_index(drop=True)
        val_img_ids = df_val['id'].tolist()
        val_labels = df_val['has_cactus'].values
        val_ds = CactusDataset(val_img_ids, val_labels, id2path=train_id2path, transforms=get_transforms("val"))
        val_loader = get_dataloader(val_ds, BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
        fold_model_path = os.path.join(MODEL_DIR, f"efficientnet_b3_fold{fold}.pt")
        model = get_efficientnet_b3(dropout_rate=dropout_rate)
        model.load_state_dict(torch.load(fold_model_path, map_location='cpu'))
        model.to(DEVICE)
        model.eval()
        fold_class_weights = class_weights if need_weights else None
        if fold_class_weights is not None:
            fold_class_weights = torch.tensor(fold_class_weights).float().to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        _, val_true, val_pred = eval_model(model, loss_fn, val_loader, DEVICE, fold_class_weights)
        val_auc = roc_auc_score(val_true, val_pred)
        oof_true.append(val_true)
        oof_pred.append(val_pred)
        fold_val_ids.append(val_img_ids)
        fold_scores.append(val_auc)
        print(f"Reloaded fold {fold}, OOF Validation AUC={val_auc:.5f}")

    all_oof_true = np.concatenate(oof_true)
    all_oof_pred = np.concatenate(oof_pred)
    oof_auc = roc_auc_score(all_oof_true, all_oof_pred)
    oof_cm = confusion_info(all_oof_true, all_oof_pred)
    print(f"OOF ROC-AUC (from loaded models): {oof_auc:.5f}")
    print(f"OOF Confusion Matrix:\n{oof_cm}")

    test_ds = CactusDataset(
        test_img_ids, labels=None,
        id2path=test_id2path,
        transforms=get_transforms("val")
    )
    test_loader = get_dataloader(test_ds, BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
    test_pred_list = []
    for fold in range(cv_fold):
        fold_model_path = os.path.join(MODEL_DIR, f"efficientnet_b3_fold{fold}.pt")
        model = get_efficientnet_b3(dropout_rate=dropout_rate)
        model.load_state_dict(torch.load(fold_model_path, map_location='cpu'))
        model.to(DEVICE)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                images, img_ids = batch
                images = images.to(DEVICE)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                preds.append(probs)
        fold_test_pred = np.concatenate(preds)
        test_pred_list.append(fold_test_pred)
        print(f"Loaded fold {fold} for test prediction.")
    test_probs = np.mean(test_pred_list, axis=0)

    submission = pd.read_csv(SAMPLE_SUB_PATH)
    submission['has_cactus'] = test_probs
    submission.to_csv('submission.csv', index=False)
    print(f"Saved submission.csv in required format with {len(submission)} rows.")

    scores_df = pd.DataFrame({
        'Model': [f"efficientnet_b3_fold{f}" for f in range(cv_fold)] + ['ensemble'],
        'ROC-AUC': list(fold_scores) + [oof_auc]
    })
    scores_df.set_index('Model', inplace=True)
    scores_df.to_csv("scores.csv")
    print(f"Saved cross-validation scores to scores.csv")

print("Section: Data Loading and Preprocessing")
try:
    train_df = pd.read_csv(TRAIN_CSV)
except Exception as e:
    print(f"Failed to load train.csv: {e}")
    sys.exit(1)
print_eda(train_df)

train_id2path = {img_id: os.path.join(TRAIN_DIR, img_id) for img_id in train_df['id']}
try:
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
except Exception as e:
    print(f"Failed to load sample_submission.csv: {e}")
    sys.exit(1)
test_img_ids = list(sample_sub['id'])
test_id2path = {img_id: os.path.join(TEST_DIR, img_id) for img_id in test_img_ids}
print(f"Loaded {len(train_id2path)} train images, {len(test_id2path)} test images.")

y_train = train_df['has_cactus'].values
class_weights, n_pos, n_neg, imbalance_ratio, need_weights = compute_class_weight(y_train)
print(f"Class stats: Pos={n_pos}, Neg={n_neg}, Imbalance Ratio(majority/minority)={imbalance_ratio:.3f}")
print(f"Use class weights: {need_weights}, Class weights: {class_weights if class_weights is not None else '[1.0,1.0]'}")
if class_weights is not None:
    np.save(os.path.join(MODEL_DIR, "class_weights.npy"), class_weights)

print("Section: Feature Engineering")
train_df = train_df.copy()
cv_fold = 5
skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=SEED)
folds = np.zeros(len(train_df), dtype=np.int32)
for idx, (_, val_idx) in enumerate(skf.split(train_df['id'], train_df['has_cactus'])):
    folds[val_idx] = idx
train_df['fold'] = folds
print(f"Assigned stratified {cv_fold}-fold indices. Fold sample counts:")
for f in range(cv_fold):
    dist = train_df.loc[train_df['fold'] == f, 'has_cactus'].value_counts().to_dict()
    print(f"  Fold {f}: n={len(train_df[train_df['fold'] == f])} class dist={dist}")

print("Section: Model Training and Evaluation")
dropout_rate = round(random.uniform(0.2, 0.5), 2)
print(f"Model config: EfficientNet-B3, Image size 300, Head dropout={dropout_rate}")

if DEBUG:
    print("DEBUG mode: using 10% subsample and 1 epoch (per fold)")
    sample_frac = 0.10
    sampled_idxs = []
    for f in range(cv_fold):
        fold_idx = train_df.index[train_df['fold'] == f].tolist()
        fold_labels = train_df.loc[fold_idx, 'has_cactus'].values
        idx_pos = [i for i, l in zip(fold_idx, fold_labels) if l == 1]
        idx_neg = [i for i, l in zip(fold_idx, fold_labels) if l == 0]
        n_pos = max(1, int(sample_frac * len(idx_pos)))
        n_neg = max(1, int(sample_frac * len(idx_neg)))
        if len(idx_pos) > 0:
            sampled_idxs += np.random.choice(idx_pos, n_pos, replace=False).tolist()
        if len(idx_neg) > 0:
            sampled_idxs += np.random.choice(idx_neg, n_neg, replace=False).tolist()
    train_df = train_df.loc[sampled_idxs].reset_index(drop=True)
    print(f"DEBUG subsample shape: {train_df.shape}")
    debug_epochs = 1
else:
    debug_epochs = None

BATCH_SIZE = 64 if torch.cuda.is_available() else 32
N_WORKERS = 4 if torch.cuda.is_available() else 1
EPOCHS = 20 if not DEBUG else debug_epochs
MIN_EPOCHS = 5 if not DEBUG else 1
EARLY_STOP_PATIENCE = 7 if not DEBUG else 2
LR = 1e-3

model_files = [os.path.join(MODEL_DIR, f"efficientnet_b3_fold{f}.pt") for f in range(cv_fold)]
if all([os.path.exists(f) for f in model_files]):
    print("All fold models found in models/. Running inference and file saving only (no retrain).")
    inference_and_submission(train_df, train_id2path, test_img_ids, test_id2path, dropout_rate,
                            class_weights, need_weights, BATCH_SIZE, N_WORKERS, cv_fold)
    return

oof_true, oof_pred, fold_scores, fold_val_ids = [], [], [], []
start_time = time.time() if DEBUG else None

for fold in range(cv_fold):
    print(f"\n=== FOLD {fold} TRAINING ===")
    df_train = train_df[train_df['fold'] != fold].reset_index(drop=True)
    df_val = train_df[train_df['fold'] == fold].reset_index(drop=True)
    print(f"Train size: {df_train.shape[0]}, Val size: {df_val.shape[0]}")
    train_img_ids = df_train['id'].tolist()
    train_labels = df_train['has_cactus'].values
    val_img_ids = df_val['id'].tolist()
    val_labels = df_val['has_cactus'].values

    train_ds = CactusDataset(
        train_img_ids, train_labels,
        id2path=train_id2path,
        transforms=get_transforms("train")
    )
    val_ds = CactusDataset(
        val_img_ids, val_labels,
        id2path=train_id2path,
        transforms=get_transforms("val")
    )
    train_loader = get_dataloader(train_ds, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    val_loader = get_dataloader(val_ds, BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
    model = get_efficientnet_b3(dropout_rate=dropout_rate)
    model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    fold_class_weights = class_weights if need_weights else None
    if fold_class_weights is not None:
        fold_class_weights = torch.tensor(fold_class_weights).float().to(DEVICE)
    best_auc = -np.inf
    best_epoch = -1
    best_model_state = None
    patience = 0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model, loss_fn, optimizer, scheduler, train_loader, DEVICE, fold_class_weights)
        val_loss, val_true, val_pred = eval_model(
            model, loss_fn, val_loader, DEVICE, fold_class_weights)
        val_auc = roc_auc_score(val_true, val_pred)
        cm = confusion_info(val_true, val_pred)
        print(f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}")
        print(f" Val confusion_matrix (rows:true [0,1]; cols:pred [0,1]):\n{cm}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if DEBUG and epoch + 1 >= debug_epochs:
            break
        if (epoch + 1) >= MIN_EPOCHS and patience >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}, best_epoch={best_epoch+1}.")
            break

    model.load_state_dict(best_model_state)
    fold_model_path = os.path.join(MODEL_DIR, f"efficientnet_b3_fold{fold}.pt")
    torch.save(model.state_dict(), fold_model_path)
    print(f"Saved best model for fold {fold} at {fold_model_path} (best_auc={best_auc:.5f}, best_epoch={best_epoch+1})")

    _, val_true, val_pred = eval_model(model, loss_fn, val_loader, DEVICE, fold_class_weights)
    oof_true.append(val_true)
    oof_pred.append(val_pred)
    fold_val_ids.append(val_img_ids)
    fold_scores.append(best_auc)
    print(f"OOF stored for fold {fold}, Validation AUC={best_auc:.5f}")

end_time = time.time() if DEBUG else None
if DEBUG:
    debug_time = end_time - start_time
    estimated_time = (1 / 0.1) * (EPOCHS / debug_epochs) * debug_time
    print("=== Start of Debug Information ===")
    print(f"debug_time: {debug_time:.1f}")
    print(f"estimated_time: {estimated_time:.1f}")
    print("=== End of Debug Information ===")

print("\nSection: Ensemble Strategy and Final Predictions")
all_oof_true = np.concatenate(oof_true)
all_oof_pred = np.concatenate(oof_pred)
oof_auc = roc_auc_score(all_oof_true, all_oof_pred)
oof_cm = confusion_info(all_oof_true, all_oof_pred)
print(f"OOF ROC-AUC: {oof_auc:.5f}")
print(f"OOF Confusion Matrix:\n{oof_cm}")

test_ds = CactusDataset(
    test_img_ids, labels=None,
    id2path=test_id2path,
    transforms=get_transforms("val")
)
test_loader = get_dataloader(test_ds, BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
test_pred_list = []
for fold in range(cv_fold):
    fold_model_path = os.path.join(MODEL_DIR, f"efficientnet_b3_fold{fold}.pt")
    model = get_efficientnet_b3(dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(fold_model_path, map_location='cpu'))
    model.to(DEVICE)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            images, img_ids = batch
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            preds.append(probs)
    fold_test_pred = np.concatenate(preds)
    test_pred_list.append(fold_test_pred)
    print(f"Loaded fold {fold} for test prediction.")
test_probs = np.mean(test_pred_list, axis=0)

print("Section: Submission File Generation")
submission = pd.read_csv(SAMPLE_SUB_PATH)
submission['has_cactus'] = test_probs
submission.to_csv('submission.csv', index=False)
print(f"Saved submission.csv in required format with {len(submission)} rows.")

scores_df = pd.DataFrame({
    'Model': [f"efficientnet_b3_fold{f}" for f in range(cv_fold)] + ['ensemble'],
    'ROC-AUC': list(fold_scores) + [oof_auc]
})
scores_df.set_index('Model', inplace=True)
scores_df.to_csv("scores.csv")
print(f"Saved cross-validation scores to scores.csv")
