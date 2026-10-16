#!/usr/bin/env python3
"""
Chl_Norm.py — Restore the chl_norm_value of PW/HZ samples from all-zeros to
the true log-normalized values

Normalization formula (consistent with RH/RB):
    norm = clip((log10(chl) - (-2.0)) / 4.0, 0, 1)
    i.e. Chl in [0.01, 100] mg/m3 -> [0, 1]

Usage:
    python Chl_Norm.py <directory>               # process the given directory
    python Chl_Norm.py <directory> --dry-run     # preview only, do not write
    python Chl_Norm.py H:\\250mOCDataset\\Train   # process the training set
    python Chl_Norm.py H:\\250mOCDataset          # process all subdirectories recursively
"""
import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path


def normalize_chl(chl_ori):
    """
    Log-normalize raw Chl values.

    Args:
        chl_ori: (128, 128) float32, raw Chl concentration (mg/m3)

    Returns:
        (128, 128) float32, normalized values in [0, 1]
    """
    chl = chl_ori.astype(np.float64).copy()

    # Handle invalid values: NaN / Inf / <=0 -> normalized to 0
    invalid = ~np.isfinite(chl) | (chl <= 0)
    chl[invalid] = 0.01  # set to the lower bound, log10(0.01) = -2 -> norm = 0

    # Log normalization: clip((log10(chl) + 2.0) / 4.0, 0, 1)
    norm = (np.log10(chl) + 2.0) / 4.0
    norm = np.clip(norm, 0.0, 1.0)

    # Force invalid pixels to 0
    norm[invalid] = 0.0

    return norm.astype(np.float32)


def process_file(filepath, dry_run=False):
    """
    Process a single .npz file: overwrite the chl_norm_value field.

    Returns:
        dict: processing statistics
    """
    # Use a context manager to ensure the file handle is closed (on Windows,
    # os.replace requires the source file to be released)
    with np.load(filepath, allow_pickle=True) as data:
        keys = list(data.keys())

        if 'chl_ori_value' not in keys:
            return {'status': 'skip', 'reason': 'missing chl_ori_value'}

        if 'chl_norm_value' not in keys:
            return {'status': 'skip', 'reason': 'missing chl_norm_value'}

        # Explicitly copy all arrays into memory to avoid NpzFile lazy loading
        # keeping the file handle open
        chl_ori = data['chl_ori_value'].astype(np.float32).copy()      # (128, 128)
        old_norm = data['chl_norm_value'].astype(np.float32).copy()     # (128, 128)

        if not dry_run:
            # Pre-load all fields into memory
            save_dict = {}
            for k in keys:
                if k == 'chl_norm_value':
                    save_dict[k] = None  # placeholder, replaced below
                else:
                    save_dict[k] = data[k].copy()

    # --- File handle closed; the following operations run in memory ---

    # Compute the new normalized values
    new_norm = normalize_chl(chl_ori)

    # Compute change statistics
    changed_pixels = np.sum(old_norm != new_norm)
    total_pixels = old_norm.size
    old_nonzero = np.sum(old_norm > 0)
    new_nonzero = np.sum(new_norm > 0)

    stats = {
        'status': 'ok',
        'total_pixels': total_pixels,
        'changed_pixels': int(changed_pixels),
        'old_nonzero': int(old_nonzero),
        'new_nonzero': int(new_nonzero),
        'new_min': float(new_norm.min()),
        'new_max': float(new_norm.max()),
        'new_mean': float(new_norm[new_norm > 0].mean()) if new_nonzero > 0 else 0.0,
    }

    if not dry_run:
        # Overwrite chl_norm_value
        save_dict['chl_norm_value'] = new_norm

        # Write a temporary file first and then replace, to avoid data
        # corruption if the write fails
        # Note: np.savez automatically appends .npz to filenames that do not
        # end with .npz, so the temporary filename must end with .npz,
        # otherwise the actual saved path would get an extra .npz suffix
        tmp_path = filepath.replace('.npz', '_tmp.npz')
        np.savez(tmp_path, **save_dict)
        os.replace(tmp_path, filepath)

    return stats


def process_directory(dirpath, dry_run=False):
    """
    Process all PW_*.npz and HZ_*.npz files under the given directory.

    Returns:
        dict: summary statistics
    """
    # Match files with the PW and HZ prefixes, recursively (Train/Val/Test subdirs)
    patterns = ['**/PW_*.npz', '**/HZ_*.npz']
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(dirpath, pat), recursive=True))

    files = sorted(files)
    total = len(files)

    if total == 0:
        print(f'[WARNING] No PW/HZ sample files found under {dirpath}')
        return {'total': 0, 'processed': 0, 'skipped': 0}

    print(f'Directory: {dirpath}')
    print(f'Matched PW/HZ files: {total}')
    print(f'Mode: {"preview (no write)" if dry_run else "write"}')
    print('-' * 70)

    processed = 0
    skipped = 0
    all_stats = {'PW': [], 'HZ': []}

    for i, fp in enumerate(files):
        fname = os.path.basename(fp)
        prefix = fname.split('_')[0]  # PW or HZ

        stats = process_file(fp, dry_run=dry_run)

        if stats['status'] == 'skip':
            skipped += 1
            print(f'  [{i+1}/{total}] SKIP  {fname} — {stats["reason"]}')
        else:
            processed += 1
            all_stats[prefix].append(stats)
            print(f'  [{i+1}/{total}] {"DRY " if dry_run else "OK  "} {prefix}  {fname}  '
                  f'changed={stats["changed_pixels"]}/{stats["total_pixels"]}  '
                  f'range=[{stats["new_min"]:.3f}, {stats["new_max"]:.3f}]  '
                  f'mean={stats["new_mean"]:.3f}')

    # Summary
    print('-' * 70)
    print(f'Done: processed {processed}, skipped {skipped}')

    for prefix in ['PW', 'HZ']:
        slist = all_stats[prefix]
        if slist:
            n = len(slist)
            total_changed = sum(s['changed_pixels'] for s in slist)
            total_pixels = sum(s['total_pixels'] for s in slist)
            all_means = [s['new_mean'] for s in slist if s['new_mean'] > 0]
            avg_mean = np.mean(all_means) if all_means else 0.0
            print(f'  {prefix}: {n} files, '
                  f'changed pixels {total_changed}/{total_pixels} ({total_changed/total_pixels*100:.1f}%), '
                  f'average normalized value {avg_mean:.4f}')

    return {'total': total, 'processed': processed, 'skipped': skipped}


def main():
    parser = argparse.ArgumentParser(
        description='Restore the chl_norm_value of PW/HZ samples from all-zeros to the true log-normalized values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python Chl_Norm.py H:\\250mOCDataset\\Train
  python Chl_Norm.py H:\\250mOCDataset\\Train --dry-run
  python Chl_Norm.py H:\\250mOCDataset\\Val
  python Chl_Norm.py H:\\250mOCDataset\\Test
        ''')
    parser.add_argument('directory', type=str, help='Directory path containing the .npz samples')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only preview the changes without actually writing files')

    args = parser.parse_args()

    dirpath = args.directory
    if not os.path.isdir(dirpath):
        print(f'[ERROR] Directory does not exist: {dirpath}')
        sys.exit(1)

    process_directory(dirpath, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
