#!/usr/bin/env python3
"""
train_baseline.py -- U-Net pixel-wise Chl-a regression baseline: training +
dual-reference evaluation (Scientific Data dataset paper).

Training labels (--label):
  ori    Original GOCI-2 Chl (GDPS L2 retrieval)
  calib  SBC (systematic bias correction) Chl (ChlCalib)

Automatic evaluation after training (depends on the label group):
  --label ori    NSOAS independent reference only (4 km merged product,
                 GOCI-NSOAS overlap >= 70% of pixels)
                 -> results/unet_ori_nsoas.json    (Group A of paper Table 4)
  --label calib  Dual references:
                 -> results/unet_calib_nsoas.json  (NSOAS independent reference,
                                                    Group B of paper Table 4)
                 -> results/unet_calib_self.json   (same-label reference:
                                                    ChlCalib train -> ChlCalib test,
                                                    2024 temporal extrapolation,
                                                    internal learnability)

Weights and results are stored separately:
  weights/  <model>_<label>.pth
  results/  <model>_<label>_nsoas.json   (generated for both ori and calib)
  results/  <model>_<label>_self.json    (calib only)

Usage (server, U-Net only):
  cd /usr/zhu_xl/ACSample20260701/SciData
  python train_baseline.py --label ori   --data_root /usr/zhu_xl/ACSample20260701
  python train_baseline.py --label calib --data_root /usr/zhu_xl/ACSample20260701

Dependencies: torch, numpy (U-Net only, no timm required).
"""
import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# ========== Configurable paths ==========
DEFAULT_DATA_ROOT = r'/usr/zhu_xl/ACSample20260701'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHT_DIR = os.path.join(SCRIPT_DIR, 'weights')
DEFAULT_RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')

TYPES = ['RH', 'RB', 'PW', 'HZ']
CLASS_LABELS = ['High (>=7)', 'Mid (2.5-7)', 'Low (1-2.5)', 'Clean (<1)']


# ========== Dataset ==========
class OCDataset(Dataset):
    def __init__(self, split, data_root, label_type='ori'):
        self.label_type = label_type
        pattern = os.path.join(data_root, split, '*.npz')
        self.files = sorted(glob.glob(pattern))
        print(f'[{split}] Loaded {len(self.files)} samples', flush=True)
        if len(self.files) == 0:
            raise RuntimeError(f'No .npz files found in {pattern}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=True) as data:
            x = torch.from_numpy(data['ac_norm_data'].copy()).float()
            if self.label_type == 'ori':
                y = torch.from_numpy(data['chl_norm_value'].copy()).float().unsqueeze(0)
            else:
                y = torch.from_numpy(data['ChlCalib_Norm_Value'].copy()).float().unsqueeze(0)
            # Same as train_baseline_v2.py: both ori and calib use Chl_Calib_Mask
            # (the dataset's valid-retrieval mask) so that the NSOAS-reference
            # results reproduce the numbers in paper Table 4.
            mask = torch.from_numpy(data['Chl_Calib_Mask'].copy()).float().unsqueeze(0)
            nsoas = torch.from_numpy(data['NSOAS_Chl_Value'].copy()).float().unsqueeze(0)
            nsoas_mask = torch.from_numpy(data['NSOAS_Valid_Mask'].copy()).float().unsqueeze(0)
            return x, y, mask, nsoas, nsoas_mask


# ========== Model (U-Net only) ==========
class UNet(nn.Module):
    """U-Net pixel-wise regression"""
    def __init__(self, in_channels=12):
        super().__init__()
        self.enc1 = self.block(in_channels, 64)
        self.enc2 = self.block(64, 128)
        self.enc3 = self.block(128, 256)
        self.enc4 = self.block(256, 512)
        self.bottleneck = self.block(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.block(128, 64)
        self.out = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.pool = nn.MaxPool2d(2)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def build_model(name):
    """Build a model by name (only U-Net is supported in this baseline)."""
    if name == 'unet':
        return UNet()
    raise ValueError(f'Unknown model: {name!r} (only "unet" is supported)')


# ========== Denormalization ==========
def denorm_chl(norm_val):
    return 10.0 ** (norm_val * 4.0 - 2.0)


# ========== Standard regression metrics (IOCCG/SeaBASS) ==========
def compute_standard_metrics(pred, ref):
    ok = np.isfinite(pred) & np.isfinite(ref) & (ref > 0) & (pred > 0)
    p, r = pred[ok], ref[ok]
    if len(p) < 2:
        return {'rmse': None, 'mae': None, 'bias': None, 'r2': None,
                'mdpd': None, 'si': None,
                'log_rmse': None, 'log_bias': None, 'log_r2': None, 'n': len(p)}
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    mae = float(np.mean(np.abs(p - r)))
    bias = float(np.mean(p - r))
    ss_res = np.sum((p - r) ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    mdpd = float(np.median(np.abs((p - r) / r) * 100))
    si = float(rmse / np.mean(r) * 100) if np.mean(r) > 0 else None
    log_p, log_r = np.log10(p), np.log10(r)
    log_rmse = float(np.sqrt(np.mean((log_p - log_r) ** 2)))
    log_bias = float(np.mean(log_p - log_r))
    log_ss_res = np.sum((log_p - log_r) ** 2)
    log_ss_tot = np.sum((log_r - np.mean(log_r)) ** 2)
    log_r2 = float(1 - log_ss_res / log_ss_tot) if log_ss_tot > 0 else None
    return {'rmse': round(rmse, 4), 'mae': round(mae, 4), 'bias': round(bias, 4),
            'r2': round(r2, 4) if r2 is not None else None,
            'mdpd': round(mdpd, 2), 'si': round(si, 2) if si is not None else None,
            'log_rmse': round(log_rmse, 4), 'log_bias': round(log_bias, 4),
            'log_r2': round(log_r2, 4) if log_r2 is not None else None,
            'n': len(p)}


# ========== Training ==========
def train_one(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, Model: {args.model}, Label: {args.label}')

    if args.eval_only:
        test_ds  = OCDataset('Test', args.data_root, args.label)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        train_loader = val_loader = None
    else:
        train_ds = OCDataset('Train', args.data_root, args.label)
        val_ds   = OCDataset('Val',   args.data_root, args.label)
        test_ds  = OCDataset('Test',  args.data_root, args.label)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet().to(device)

    if args.eval_only:
        # Existing weights: skip training, load and evaluate directly
        assert args.ckpt and os.path.exists(args.ckpt), \
            f'--eval_only requires an existing --ckpt, got {args.ckpt!r}'
        best_state = torch.load(args.ckpt, map_location=device)
        print(f'Eval-only mode: checkpoint loaded from {args.ckpt}')
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_loss, best_state = float('inf'), None
        for epoch in range(args.epochs):
            model.train()
            t_loss, t_pix = 0, 0
            for x, y, mask, _, _ in train_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = ((pred - y) ** 2 * mask).sum() / (mask.sum() + 1e-8)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * mask.sum().item()
                t_pix  += mask.sum().item()

            model.eval()
            v_loss, v_pix = 0, 0
            with torch.no_grad():
                for x, y, mask, _, _ in val_loader:
                    x, y, mask = x.to(device), y.to(device), mask.to(device)
                    pred = model(x)
                    v_loss += ((pred - y) ** 2 * mask).sum().item()
                    v_pix  += mask.sum().item()
            print(f'Epoch {epoch+1}/{args.epochs}: train_rmse_norm={np.sqrt(t_loss/max(t_pix,1)):.4f}, '
                  f'val_rmse_norm={np.sqrt(v_loss/max(v_pix,1)):.4f}', flush=True)
            scheduler.step()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # ===== Save weights (weights/) =====
        os.makedirs(args.weight_dir, exist_ok=True)
        model_path = os.path.join(args.weight_dir, f'{args.model}_{args.label}.pth')
        if best_state is not None:
            torch.save(best_state, model_path)
            print(f'Model saved to {model_path}')

    # ===== Dual-reference evaluation =====
    model.load_state_dict(best_state)
    model.eval()

    thresh = args.overlap_thresh
    # Two collectors: NSOAS reference / same-label reference
    nsoas_agg = {t: {'pred': [], 'ref': []} for t in TYPES}
    self_agg  = {t: {'pred': [], 'ref': []} for t in TYPES}
    all_nsoas_pred, all_nsoas_ref = [], []
    all_self_pred,  all_self_ref  = [], []
    unified_cm_nsoas = np.zeros((4, 4), dtype=np.int64)
    unified_cm_self  = np.zeros((4, 4), dtype=np.int64)
    t_kept_nsoas = t_kept_self = s_total = 0

    with torch.no_grad():
        for x, y, mask, nsoas, nsm in test_loader:
            x = x.to(device)
            pred_norm = model(x)
            # Drop the single-channel dim ((1,H,W) -> (H,W) per sample within the
            # batch) to match the mask shape
            pred_chl  = denorm_chl(pred_norm.cpu().numpy())[:, 0]
            ref_self  = denorm_chl(y.numpy())[:, 0]     # same-label reference (same convention as the training label)
            nsoas_chl = nsoas.numpy()[:, 0]
            gv = mask.numpy().astype(bool)[:, 0]
            nv = nsm.numpy().astype(bool)[:, 0]

            for i in range(x.size(0)):
                fname = os.path.basename(test_ds.files[s_total]) if s_total < len(test_ds.files) else ''
                stype = fname.split('_')[0]
                if stype not in self_agg:
                    continue
                s_total += 1

                g, n = gv[i], nv[i]
                ov = g & n
                ng = g.sum()
                if ng == 0:
                    continue

                # ---- NSOAS reference (overlap >= 70%) ----
                if ov.sum() / ng >= thresh:
                    no = ov.sum()
                    if no > 0:
                        pp = pred_chl[i][ov]
                        rr = nsoas_chl[i][ov]
                        v = np.isfinite(pp) & (pp > 0) & np.isfinite(rr) & (rr > 0)
                        if v.sum() > 0:
                            nsoas_agg[stype]['pred'].extend(pp[v].tolist())
                            nsoas_agg[stype]['ref'].extend(rr[v].tolist())
                            all_nsoas_pred.extend(pp[v].tolist())
                            all_nsoas_ref.extend(rr[v].tolist())
                            t_kept_nsoas += 1
                            _accum_cm(unified_cm_nsoas, nsoas_chl[i], pred_chl[i], ov)

                # ---- Same-label reference (calib group only; all pixels inside
                #      Chl_Calib_Mask) ----
                # Note: the ori group has no same-label evaluation -- the ori label
                # is a deterministic output of the GDPS algorithm, so a Group-A
                # same-label evaluation would degenerate into "fitting the GDPS
                # algorithm" and has no independent scientific value.
                if args.label == 'calib':
                    sv = g
                    if sv.sum() > 0:
                        pp = pred_chl[i][sv]
                        rr = ref_self[i][sv]
                        v = np.isfinite(pp) & (pp > 0) & np.isfinite(rr) & (rr > 0)
                        if v.sum() > 0:
                            self_agg[stype]['pred'].extend(pp[v].tolist())
                            self_agg[stype]['ref'].extend(rr[v].tolist())
                            all_self_pred.extend(pp[v].tolist())
                            all_self_ref.extend(rr[v].tolist())
                            t_kept_self += 1
                            _accum_cm(unified_cm_self, ref_self[i], pred_chl[i], sv)

    # ===== Result summary =====
    res_dir = args.result_dir
    os.makedirs(res_dir, exist_ok=True)
    ts = datetime.now().isoformat()
    base = {'model': args.model, 'label_type': args.label, 'timestamp': ts}

    # NSOAS reference
    res_nsoas = dict(base, reference='nsoas')
    res_nsoas['overall'] = compute_standard_metrics(np.array(all_nsoas_pred), np.array(all_nsoas_ref))
    res_nsoas['per_type'] = {t: compute_standard_metrics(np.array(nsoas_agg[t]['pred']), np.array(nsoas_agg[t]['ref']))
                             for t in TYPES}
    res_nsoas['classification'] = cls_from_cm(unified_cm_nsoas)
    res_nsoas['n_kept'] = t_kept_nsoas
    res_nsoas['n_samples'] = s_total
    fn_nsoas = os.path.join(res_dir, f'{args.model}_{args.label}_nsoas.json')
    with open(fn_nsoas, 'w') as f:
        json.dump(res_nsoas, f, indent=2)

    # Same-label reference (calib group only)
    res_self = None
    fn_self = None
    if args.label == 'calib':
        res_self = dict(base, reference='same_label')
        res_self['overall'] = compute_standard_metrics(np.array(all_self_pred), np.array(all_self_ref))
        res_self['per_type'] = {t: compute_standard_metrics(np.array(self_agg[t]['pred']), np.array(self_agg[t]['ref']))
                                for t in TYPES}
        res_self['classification'] = cls_from_cm(unified_cm_self)
        res_self['n_kept'] = t_kept_self
        res_self['n_samples'] = s_total
        fn_self = os.path.join(res_dir, f'{args.model}_{args.label}_self.json')
        with open(fn_self, 'w') as f:
            json.dump(res_self, f, indent=2)

    # ===== Print comparison =====
    print(f'\n=== {args.model} / {args.label} ===')
    print(f'Samples: {s_total} | NSOAS-kept: {t_kept_nsoas}'
          + (f' | same-label-kept: {t_kept_self}' if args.label == 'calib' else ''))
    entries = [('NSOAS reference', res_nsoas)]
    if res_self is not None:
        entries.append(('same-label reference', res_self))
    for name, r in entries:
        m = r['overall']
        print(f'\n--- {name} ---')
        print(f'  Linear: RMSE={m["rmse"]} MAE={m["mae"]} Bias={m["bias"]} R2={m["r2"]} '
              f'MDPD={m["mdpd"]}% SI={m["si"]}%')
        print(f'  Log10:  RMSE={m["log_rmse"]} Bias={m["log_bias"]} R2={m["log_r2"]}')
        c = r['classification']
        print(f'  Class:  Accuracy={c["accuracy"]} Kappa={c["kappa"]} Macro-F1={c["macro_f1"]}')
    print(f'\nSaved: {fn_nsoas}'
          + (f'\n       {fn_self}' if fn_self else ''))


def _accum_cm(cm, ref_chl, pred_chl, valid_pix):
    """Accumulate the four-class threshold-classification confusion matrix
    (unified thresholds: High>=7, Mid 2.5-7, Low 1-2.5, Clean<1)."""
    vp = valid_pix & np.isfinite(ref_chl) & np.isfinite(pred_chl) & (ref_chl > 0) & (pred_chl > 0)
    if vp.sum() == 0:
        return
    gt = np.full(ref_chl.shape, -1, dtype=int)
    gt[(ref_chl >= 7) & vp] = 0
    gt[(ref_chl >= 2.5) & (ref_chl < 7) & vp] = 1
    gt[(ref_chl >= 1) & (ref_chl < 2.5) & vp] = 2
    gt[(ref_chl < 1) & vp] = 3
    pd = np.full(pred_chl.shape, -1, dtype=int)
    pd[(pred_chl >= 7) & vp] = 0
    pd[(pred_chl >= 2.5) & (pred_chl < 7) & vp] = 1
    pd[(pred_chl >= 1) & (pred_chl < 2.5) & vp] = 2
    pd[(pred_chl < 1) & vp] = 3
    vm = (gt >= 0) & (pd >= 0)
    for g, p in zip(gt[vm], pd[vm]):
        cm[g, p] += 1


def cls_from_cm(cm):
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = float(correct / total) if total > 0 else 0.0
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / total
    kappa = float((correct - np.trace(expected)) / (total - np.trace(expected))) if total > np.trace(expected) else 0.0
    per_class = {}
    for i, label in enumerate(CLASS_LABELS):
        tp_i = cm[i, i]
        fp_i = cm[:, i].sum() - tp_i
        fn_i = cm[i, :].sum() - tp_i
        p_i = float(tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else 0.0
        r_i = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
        f1_i = float(2 * p_i * r_i / (p_i + r_i)) if (p_i + r_i) > 0 else 0.0
        per_class[label] = {'precision': round(p_i, 4), 'recall': round(r_i, 4), 'f1': round(f1_i, 4)}
    macro_f1 = float(np.mean([per_class[l]['f1'] for l in CLASS_LABELS]))
    return {'accuracy': round(accuracy, 4), 'kappa': round(kappa, 4),
            'macro_f1': round(macro_f1, 4), 'per_class': per_class,
            'confusion_matrix': cm.tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U-Net baseline: train + dual-reference evaluation')
    parser.add_argument('--model', default='unet', choices=['unet'], help='U-Net only')
    parser.add_argument('--label', required=True, choices=['ori', 'calib'])
    parser.add_argument('--overlap_thresh', type=float, default=0.7)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--weight_dir', type=str, default=DEFAULT_WEIGHT_DIR)
    parser.add_argument('--result_dir', type=str, default=DEFAULT_RESULT_DIR)
    parser.add_argument('--eval_only', action='store_true', help='evaluate only with existing weights, skip training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path required by --eval_only')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f'Model={args.model}, Label={args.label}, GPU={args.gpu}')
    print(f'Weights -> {args.weight_dir}\nResults -> {args.result_dir}')
    train_one(args)
