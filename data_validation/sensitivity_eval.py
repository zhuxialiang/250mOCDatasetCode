#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensitivity_eval.py -- Dataset-validation experiment evaluation / sensitivity
check (no retraining; evaluates the test set with existing weights only).

Background: after the dataset switched from H:\\250mOCDataset (linear-domain
NSOAS) to the released version H:\\250mOCDataset2 (log-domain NSOAS), the
following fields changed:
  - ChlCalib_Norm_Value (B/calib training label, p95 relative difference ~2.1%)
  - NSOAS_Chl_Value / NSOAS_Valid_Mask (independent test-set reference)
The A/ori training label chl_norm_value is unchanged.

This script evaluates the test set with [existing weights under --model_dir] +
[the new dataset], following exactly the same evaluation logic as
train_baseline.py (overlap >= 0.7 filtering, IOCCG regression metrics, the
four-class threshold-classification confusion matrix), and adds one derived
metric:
  - Binary Chl>=3 Precision/Recall/F1 (merging the >=7 and 2.5-7 rows of the
    four-class matrix) -- consistent with the SBC anchoring-region boundary at
    Chl > 3, as a supplementary robustness metric for the high-value regime.

Usage (server; pass the new dataset root via --data_root):
  python sensitivity_eval.py \
      --data_root   /usr/zhu_xl/250mOCDataset2 \
      --model_dir   /usr/zhu_xl/250mOCDataset2/ValCode/results_d2 \
      --output_dir  /usr/zhu_xl/250mOCDataset2/ValCode/results_sens \
      --gpu 0

  --model_dir must contain {model}_{label}.pth weights (optionally a .json of
  the same name for a new-vs-old comparison).
  Use --model / --label to filter the evaluation combos (all 4 by default).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_baseline as tb  # reuses OCDataset / build_model / denorm_chl / compute_standard_metrics

TYPES = ['RH', 'RB', 'PW', 'HZ']
CLASS_LABELS = ['High (>=7)', 'Mid (2.5-7)', 'Low (1-2.5)', 'Clean (<1)']
METRICS = ['rmse', 'mae', 'bias', 'r2', 'mdpd', 'si', 'log_rmse', 'log_bias', 'log_r2']


def binary_ge3(cm):
    """Derive binary Chl>=3 metrics from the 4x4 confusion matrix
    (GT and prediction both split at >=3 vs <3).

    Rows/columns of cm are the GT/Pred four classes:
    0=High(>=7), 1=Mid(2.5-7), 2=Low(1-2.5), 3=Clean(<1).
    High (>=3) = class 0 + class 1.
    """
    cm = np.asarray(cm, dtype=np.int64)
    tp = cm[:2, :2].sum()
    fp = cm[2:, :2].sum()
    fn = cm[:2, 2:].sum()
    tn = cm[2:, 2:].sum()
    p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}


def evaluate(model_name, label, weight_path, data_root, overlap_thresh,
             device, batch_size=32, num_workers=4):
    """Replicate the train_baseline.py test-set evaluation; returns the same
    structure of dict."""
    test_ds = tb.OCDataset('Test', data_root, label)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    model = tb.build_model(model_name).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    type_pred = {t: {'pred': [], 'nsoas': []} for t in TYPES}
    all_pred, all_nsoas = [], []
    t_kept, s_total = 0, 0
    unified_cm = np.zeros((4, 4), dtype=np.int64)

    with torch.no_grad():
        for x, y, mask, nsoas, nsm in test_loader:
            x = x.to(device)
            pred_norm = model(x)
            pred_chl = tb.denorm_chl(pred_norm.cpu().numpy())
            nsoas_chl = nsoas.numpy()
            nsoas_v = nsm.numpy().astype(bool)
            goci_v = mask.numpy().astype(bool)

            for i in range(x.size(0)):
                fname = os.path.basename(test_ds.files[s_total + i]) \
                    if s_total + i < len(test_ds.files) else ''
                stype = fname.split('_')[0]
                if stype not in type_pred:
                    continue
                s_total += 1
                gv, nv = goci_v[i, 0], nsoas_v[i, 0]
                ov = gv & nv
                ng = gv.sum()
                if ng == 0:
                    continue
                if ov.sum() / ng < overlap_thresh:
                    continue
                t_kept += 1
                no = ov.sum()
                if no == 0:
                    continue
                pp = pred_chl[i, 0][ov]
                nn = nsoas_chl[i, 0][ov]
                valid = np.isfinite(pp) & (pp > 0) & np.isfinite(nn) & (nn > 0)
                if valid.sum() == 0:
                    continue
                type_pred[stype]['pred'].extend(pp[valid].tolist())
                type_pred[stype]['nsoas'].extend(nn[valid].tolist())
                all_pred.extend(pp[valid].tolist())
                all_nsoas.extend(nn[valid].tolist())

                # Four-class threshold classification (same as train_baseline,
                # GT = NSOAS)
                lc = nsoas_chl[i, 0]
                pc = pred_chl[i, 0]
                valid_pix = ov & np.isfinite(lc) & np.isfinite(pc) & (lc > 0) & (pc > 0)
                if valid_pix.sum() == 0:
                    continue
                gt_class = np.full(lc.shape, -1, dtype=int)
                gt_class[(lc >= 7) & valid_pix] = 0
                gt_class[(lc >= 2.5) & (lc < 7) & valid_pix] = 1
                gt_class[(lc >= 1) & (lc < 2.5) & valid_pix] = 2
                gt_class[(lc < 1) & valid_pix] = 3
                pd_class = np.full(pc.shape, -1, dtype=int)
                pd_class[(pc >= 7) & valid_pix] = 0
                pd_class[(pc >= 2.5) & (pc < 7) & valid_pix] = 1
                pd_class[(pc >= 1) & (pc < 2.5) & valid_pix] = 2
                pd_class[(pc < 1) & valid_pix] = 3
                valid_mask = (gt_class >= 0) & (pd_class >= 0)
                for g, p in zip(gt_class[valid_mask], pd_class[valid_mask]):
                    unified_cm[g, p] += 1

    overall = tb.compute_standard_metrics(np.array(all_pred), np.array(all_nsoas))
    per_type = {t: tb.compute_standard_metrics(np.array(type_pred[t]['pred']),
                                               np.array(type_pred[t]['nsoas']))
                for t in TYPES}

    total_pixels = unified_cm.sum()
    correct = np.trace(unified_cm)
    accuracy = float(correct / total_pixels) if total_pixels > 0 else 0.0
    row_sum = unified_cm.sum(axis=1)
    col_sum = unified_cm.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / total_pixels
    kappa = float((correct - np.trace(expected)) / (total_pixels - np.trace(expected))) \
        if total_pixels > np.trace(expected) else 0.0
    per_class = {}
    for i, lab in enumerate(CLASS_LABELS):
        tp_i = unified_cm[i, i]
        fp_i = unified_cm[:, i].sum() - tp_i
        fn_i = unified_cm[i, :].sum() - tp_i
        p_i = float(tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else 0.0
        r_i = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
        f1_i = float(2 * p_i * r_i / (p_i + r_i)) if (p_i + r_i) > 0 else 0.0
        per_class[lab] = {'precision': round(p_i, 4), 'recall': round(r_i, 4),
                          'f1': round(f1_i, 4)}
    macro_f1 = float(np.mean([per_class[l]['f1'] for l in CLASS_LABELS]))
    ge3 = binary_ge3(unified_cm)

    return {
        'model': model_name, 'label_type': label,
        'kept_total': [t_kept, s_total],
        'overall': overall,
        'per_type': per_type,
        'classification': {
            'accuracy': round(accuracy, 4), 'kappa': round(kappa, 4),
            'macro_f1': round(macro_f1, 4), 'per_class': per_class,
            'binary_ge3': ge3,
            'confusion_matrix': unified_cm.tolist(),
        },
    }


def fmt(v):
    if v is None:
        return '  n/a '
    if isinstance(v, (int, float)):
        return f'{v:8.4f}'
    return str(v)


def diff(new_v, old_v):
    if new_v is None or old_v is None or not isinstance(new_v, (int, float)):
        return '     '
    return f'{new_v - old_v:+8.4f}'


def main():
    ap = argparse.ArgumentParser(description='Evaluate the test set with existing weights and the new dataset')
    ap.add_argument('--data_root', required=True, help='dataset root directory (contains Train/Val/Test)')
    ap.add_argument('--model_dir', required=True, help='weight directory (contains {model}_{label}.pth; an optional .json of the same name enables the new-vs-old comparison)')
    ap.add_argument('--output_dir', default=None, help='output directory for results (default: model_dir/sensitivity)')
    ap.add_argument('--overlap_thresh', type=float, default=0.7)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--model', choices=['unet'], default=None,
                    help='evaluate only the given model (default: unet)')
    ap.add_argument('--label', choices=['ori', 'calib'], default=None,
                    help='evaluate only the given label group (default: all; group A = ori, group B = calib)')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  data_root: {args.data_root}  overlap threshold: {args.overlap_thresh}')

    out_dir = args.output_dir or os.path.join(args.model_dir, 'sensitivity')
    os.makedirs(out_dir, exist_ok=True)

    report = {}
    models = ['unet'] if args.model is None else [args.model]
    labels = ['ori', 'calib'] if args.label is None else [args.label]
    for model in models:
        for label in labels:
            weight = os.path.join(args.model_dir, f'{model}_{label}.pth')
            old_json = os.path.join(args.model_dir, f'{model}_{label}.json')
            print(f'\n{"="*64}\n{model} / {label}\n{"="*64}')
            if not os.path.exists(weight):
                print(f'  weight not found, skipping: {weight}')
                continue
            new = evaluate(model, label, weight, args.data_root, args.overlap_thresh,
                           device, args.batch_size, args.num_workers)
            old = json.load(open(old_json)) if os.path.exists(old_json) else None

            nk = new['kept_total']
            okt = (old or {}).get('kept_total')
            if okt:
                print(f'  kept samples (overlap>=70%): new {nk[0]}/{nk[1]}  old {okt[0]}/{okt[1]}  '
                      f'delta {(nk[0]-okt[0]):+d}')
            else:
                print(f'  kept samples (overlap>=70%): {nk[0]}/{nk[1]}  (no old result)')
            print(f'  {"metric":10s} {"this eval":>18s} {"old (ref)":>18s} {"delta":>9s}')
            for k in METRICS:
                nv = new['overall'].get(k)
                ov = ((old or {}).get('overall') or {}).get(k)
                print(f'  {k:10s} {fmt(nv):>18s} {fmt(ov):>18s} {diff(nv, ov):>9s}')
            nc = new['classification']
            oc = (old or {}).get('classification')
            if oc:
                print(f'  classification accuracy: new {nc["accuracy"]:.4f}  old {oc["accuracy"]:.4f}  '
                      f'delta {nc["accuracy"]-oc["accuracy"]:+.4f}')
                print(f'  macro_f1:               new {nc["macro_f1"]:.4f}  old {oc["macro_f1"]:.4f}  '
                      f'delta {nc["macro_f1"]-oc["macro_f1"]:+.4f}')
            else:
                print(f'  classification accuracy: {nc["accuracy"]:.4f}   macro_f1: {nc["macro_f1"]:.4f}')
            g3 = nc['binary_ge3']
            print(f'  binary Chl>=3:          P {g3["precision"]:.4f}  R {g3["recall"]:.4f}  '
                  f'F1 {g3["f1"]:.4f}  (TP {g3["tp"]}, FP {g3["fp"]}, FN {g3["fn"]}, TN {g3["tn"]})')
            print('  per-type log10-RMSE: ' + '  '.join(
                f'{t} {new["per_type"][t]["log_rmse"] if new["per_type"][t]["log_rmse"] is not None else "n/a":>7}'
                for t in TYPES))

            report[f'{model}_{label}'] = {'new': new, 'old': old}
            with open(os.path.join(out_dir, f'{model}_{label}_new.json'), 'w') as f:
                json.dump(new, f, indent=2, default=str)

    with open(os.path.join(out_dir, 'sensitivity_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\nNew results and the summary report written to: {out_dir}/  (sensitivity_report.json)')


if __name__ == '__main__':
    main()
