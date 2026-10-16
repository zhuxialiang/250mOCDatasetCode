#!/usr/bin/env python3
"""
eval_same_label.py -- Same-label evaluation: model performance on the test set
against the label of the same definition (train/test share the same label
convention; identical IOCCG metrics and four-class threshold classification
as paper Table 4).

Purpose: supplements the experiments not reported in the paper, i.e.
ChlCalib train -> ChlCalib test (and GOCI train -> GOCI test).

Usage (run on the training machine/server where the weights exist; takes a
few minutes):
  python eval_same_label.py --model unet --label calib --ckpt <path>/unet_calib.pth
  python eval_same_label.py --model unet --label ori   --ckpt <path>/unet_ori.pth

Arguments:
  --model      model name (only unet supports same-label evaluation; extend the
               model class yourself for other architectures)
  --label      calib (SBC-corrected label) or ori (original GOCI label)
  --ckpt       path to the trained .pth weights (required)
  --data_root  dataset root directory (contains Train/Val/Test subdirectories
               with *.npz each)
  --output     output json path (default: <script dir>/results/<model>_<label>_self.json)
  --batch_size inference batch size
  --gpu        CUDA device id (e.g. 0)
"""
import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TYPES = ['RH', 'RB', 'PW', 'HZ']
CLASS_LABELS = ['High (>=7)', 'Mid (2.5-7)', 'Low (1-2.5)', 'Clean (<1)']


# ========== Model (identical to the UNet in train_baseline.py) ==========
class UNet(nn.Module):
    """U-Net regression"""
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


# ========== Dataset ==========
class TestDataset(Dataset):
    def __init__(self, split, data_root, label_type):
        self.label_type = label_type
        pattern = os.path.join(data_root, split, '*.npz')
        self.files = sorted(glob.glob(pattern))
        print(f'[{split}] Loaded {len(self.files)} samples', flush=True)
        if len(self.files) == 0:
            raise RuntimeError(f'No .npz files found in {pattern}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = os.path.basename(self.files[idx])
        stype = fname.split('_')[0]
        with np.load(self.files[idx], allow_pickle=True) as data:
            x = torch.from_numpy(data['ac_norm_data'].copy()).float()
            if self.label_type == 'calib':
                y = torch.from_numpy(data['ChlCalib_Norm_Value'].copy()).float().unsqueeze(0)
            else:
                y = torch.from_numpy(data['chl_norm_value'].copy()).float().unsqueeze(0)
            mask = torch.from_numpy(data['Chl_Calib_Mask'].copy()).float().unsqueeze(0)
        return x, y, mask, stype


# ========== Denormalization ==========
def denorm_chl(norm_val):
    return 10.0 ** (norm_val * 4.0 - 2.0)


# ========== Metrics (identical to train_baseline.py) ==========
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


def main():
    parser = argparse.ArgumentParser(description='Same-label evaluation (train/test share the same label convention)')
    parser.add_argument('--model', required=True, choices=['unet'])
    parser.add_argument('--label', required=True, choices=['calib', 'ori'])
    parser.add_argument('--ckpt', required=True, help='path to <model>_<label>.pth')
    parser.add_argument('--data_root', default=r'H:\ACSample20260701')
    parser.add_argument('--output', default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ds = TestDataset('Test', args.data_root, args.label)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f'Checkpoint loaded: {args.ckpt}')

    ref_name = 'ChlCalib' if args.label == 'calib' else 'GOCI (ori)'
    print(f'Reference label: {ref_name}')

    type_data = {t: {'pred': [], 'ref': []} for t in TYPES}
    all_pred, all_ref = [], []
    n_kept, n_samples = 0, 0
    unified_cm = np.zeros((4, 4), dtype=np.int64)

    with torch.no_grad():
        for x, y, mask, stype in loader:
            x = x.to(device)
            pred_norm = model(x)
            pred_chl = denorm_chl(pred_norm.cpu().numpy())
            ref_chl = denorm_chl(y.numpy())
            m = mask.numpy().astype(bool)
            for i in range(x.size(0)):
                if stype[i] not in type_data:
                    continue
                n_samples += 1
                valid = m[i, 0] & np.isfinite(pred_chl[i, 0]) & np.isfinite(ref_chl[i, 0]) \
                        & (pred_chl[i, 0] > 0) & (ref_chl[i, 0] > 0)
                if valid.sum() < 2:
                    continue
                n_kept += 1
                pp = pred_chl[i, 0][valid]
                rr = ref_chl[i, 0][valid]
                type_data[stype[i]]['pred'].extend(pp.tolist())
                type_data[stype[i]]['ref'].extend(rr.tolist())
                all_pred.extend(pp.tolist())
                all_ref.extend(rr.tolist())

                # Four-class threshold classification
                lc = ref_chl[i, 0]
                pc = pred_chl[i, 0]
                vp = m[i, 0] & np.isfinite(lc) & np.isfinite(pc) & (lc > 0) & (pc > 0)
                if vp.sum() == 0:
                    continue
                gt_class = np.full(lc.shape, -1, dtype=int)
                gt_class[(lc >= 7) & vp] = 0
                gt_class[(lc >= 2.5) & (lc < 7) & vp] = 1
                gt_class[(lc >= 1) & (lc < 2.5) & vp] = 2
                gt_class[(lc < 1) & vp] = 3
                pd_class = np.full(pc.shape, -1, dtype=int)
                pd_class[(pc >= 7) & vp] = 0
                pd_class[(pc >= 2.5) & (pc < 7) & vp] = 1
                pd_class[(pc >= 1) & (pc < 2.5) & vp] = 2
                pd_class[(pc < 1) & vp] = 3
                vm = (gt_class >= 0) & (pd_class >= 0)
                for g, p in zip(gt_class[vm], pd_class[vm]):
                    unified_cm[g, p] += 1

    overall = compute_standard_metrics(np.array(all_pred), np.array(all_ref))
    per_type = {t: compute_standard_metrics(np.array(type_data[t]['pred']), np.array(type_data[t]['ref']))
                for t in TYPES}

    total = unified_cm.sum()
    correct = np.trace(unified_cm)
    accuracy = float(correct / total) if total > 0 else 0.0
    row_sum = unified_cm.sum(axis=1)
    col_sum = unified_cm.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / total
    kappa = float((correct - np.trace(expected)) / (total - np.trace(expected))) if total > np.trace(expected) else 0.0
    per_class = {}
    for i, label in enumerate(CLASS_LABELS):
        tp_i = unified_cm[i, i]
        fp_i = unified_cm[:, i].sum() - tp_i
        fn_i = unified_cm[i, :].sum() - tp_i
        p_i = float(tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else 0.0
        r_i = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
        f1_i = float(2 * p_i * r_i / (p_i + r_i)) if (p_i + r_i) > 0 else 0.0
        per_class[label] = {'precision': round(p_i, 4), 'recall': round(r_i, 4), 'f1': round(f1_i, 4)}
    macro_f1 = float(np.mean([per_class[l]['f1'] for l in CLASS_LABELS]))

    print(f'\n=== {args.model} / {args.label} / same-label ===')
    print(f'Kept: {n_kept}/{n_samples} samples')
    print(f'  Linear: RMSE={overall["rmse"]}, MAE={overall["mae"]}, Bias={overall["bias"]}, '
          f'R2={overall["r2"]}, MDPD={overall["mdpd"]}%, SI={overall["si"]}%')
    print(f'  Log10:  RMSE={overall["log_rmse"]}, Bias={overall["log_bias"]}, R2={overall["log_r2"]}')
    for t in TYPES:
        m = per_type[t]
        if m['n'] > 0 and m['rmse'] is not None:
            print(f'  {t}: logRMSE={m["log_rmse"]} logBias={m["log_bias"]} MDPD={m["mdpd"]}% SI={m["si"]}% (N={m["n"]})')
    print(f'Classification: Accuracy={accuracy:.4f} Kappa={kappa:.4f} Macro-F1={macro_f1:.4f}')

    result = {
        'model': args.model, 'label_type': args.label, 'reference': 'same_label',
        'overall': overall, 'per_type': per_type,
        'classification': {'accuracy': round(accuracy, 4), 'kappa': round(kappa, 4),
                           'macro_f1': round(macro_f1, 4), 'per_class': per_class,
                           'confusion_matrix': unified_cm.tolist()},
        'n_kept': n_kept, 'n_samples': n_samples,
    }
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'results', f'{args.model}_{args.label}_self.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
