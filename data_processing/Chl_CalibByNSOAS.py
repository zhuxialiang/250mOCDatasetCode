"""
Chl_CalibByNSOAS.py  V4 — Chl>3 anchoring + sigmoid soft-transition calibration
Radiometric baseline correction of GOCI-2 samples using the NSOAS multi-source
merged L3A Chl product.

Method:
  NSOAS 4km -> log-domain bilinear resampling (interpolate in log10 space,
  back-transform) -> GOCI 128x128 grid
  V4 (Chl>3 anchoring + sigmoid soft transition):
    1. Mask = GOCI-2 AC-valid ∩ Chl non-cloud/non-land ∩ Chl>0
    2. Anchor region: intersection of GOCI>=3 ∩ NSOAS>=3, compute the median ratio:
       scale = clip(median(NSOAS)/median(GOCI), 0.1, 10.0)
       Additional constraint: overlap ratio of NSOAS in the GOCI high-value
       region must be >= 70%
    3. Apply weight: w = sigmoid((GOCI - 3.5) / 0.5)
       Chl_Calib = GOCI x (1 + w x (scale - 1))
       - GOCI <= 3.0: w ~ 0 -> keep original value
       - GOCI = 3.5:  w = 0.5 -> half-calibrated
       - GOCI >= 4.0: w ~ 1 -> fully calibrated
    4. Hard truncation: Chl_Calib <= 100 mg/m3

Design evolution:
  V1: per-block ratio_field -> values explode in the low-Chl region, discarded
  V2: full-set intersection -> intersection GOCI biased low, scale pulled up,
      discarded
  V3: GOCI>P75 anchoring -> calibration healthy but does not separate the
      low/high value regions, discarded
  V4: Chl>3 anchoring + sigmoid transition -> calibrate only the high-value
      region, keep the low-value region unchanged, no hard boundary

Usage:
  python Chl_CalibByNSOAS.py --dir H:\250mOCDataset\Train --overwrite
  python Chl_CalibByNSOAS.py --dir H:\250mOCDataset\Val   --overwrite
  python Chl_CalibByNSOAS.py --dir H:\250mOCDataset\Test  --overwrite

Notes:
  - To re-calibrate samples that already embed NSOAS data, prefer
    ChlCalib_Norm.py (faster, no external H5 required)
  - This script is kept for calibrating brand-new samples (without NSOAS
    fields) from scratch
"""

import os, re, glob, h5py, argparse
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# ============================================================
# Local path to the NSOAS L3A multi-source merged Chl product root directory.
# IMPORTANT: this is the author's local path. Set it to your own download
# location (or pass it via --nsoas_root) before running.
NSOAS_ROOT = r"E:/NSOAS_product_root"
TARGET_SIZE = 128
CHL_MAX_VALID = 100.0    # upper bound of the GOCI-2 OC algorithm calibration domain

# V4 parameters
CHL_ANCHOR = 3.0          # Chl lower bound of the anchor region (both GOCI and NSOAS must be >= 3)
CHL_TRANSITION = 3.5      # Sigmoid center
CHL_TRANSITION_W = 0.5    # Sigmoid width
NSOAS_OVERLAP_MIN = 0.70  # minimum overlap ratio of NSOAS with GOCI in the high-value region

# Chl flag masks
CHL_MASK_CLOUD = 1
CHL_MASK_LAND = 2
CHL_INVALID = CHL_MASK_CLOUD | CHL_MASK_LAND

# GOCI-2 self-validity mask: combination of AC flag + Chl flag
AC_MASK_CLOUD = 1; AC_MASK_LAND = 2; AC_MASK_AC_FAIL = 4
AC_MASK_SUNGLINT = 16; AC_MASK_SATURATION = 32; AC_MASK_LOW_SNR = 64
AC_INVALID = (AC_MASK_CLOUD | AC_MASK_LAND | AC_MASK_AC_FAIL |
              AC_MASK_SUNGLINT | AC_MASK_SATURATION | AC_MASK_LOW_SNR)

LOG_MIN = -2.0
LOG_SPAN = 4.0


def sigmoid(x):
    """Numerically stable sigmoid function."""
    x_safe = np.clip(x, -10.0, 10.0)
    return 1.0 / (1.0 + np.exp(-x_safe))


def build_nsoas_index(root):
    """Build a NSOAS .h5 index {YYYYMMDD: path}."""
    print("Building NSOAS file index...")
    idx = {}
    for y in os.listdir(root):
        yp = os.path.join(root, y)
        if not os.path.isdir(yp) or not y.isdigit(): continue
        for m in os.listdir(yp):
            mp = os.path.join(yp, m)
            if not os.path.isdir(mp) or not m.isdigit(): continue
            for d in os.listdir(mp):
                dp = os.path.join(mp, d)
                if not os.path.isdir(dp) or not d.isdigit(): continue
                for f in os.listdir(dp):
                    if f.endswith('.h5') and '_CHL_' in f:
                        idx[y + m + d] = os.path.join(dp, f)
    print(f"  Indexed {len(idx)} days")
    return idx


def resample_to_target(src_data, src_lat, src_lon, tgt_lat, tgt_lon):
    """Log-domain bilinear interpolation: src -> tgt grid. src_lat/lon must be increasing; output row0 corresponds to tgt_lat[0].
    Interpolates in log10 space and back-transforms, avoiding the systematic
    overestimation that linear-space bilinear interpolation introduces in frontal
    regions where Chl spans orders of magnitude (arithmetic mean >= geometric
    mean); consistent with the log-normal character of Chl distributions
    (methodological validation in Supplementary Note S1.8)."""
    if src_data.size == 0:
        return np.full((len(tgt_lat), len(tgt_lon)), np.nan, dtype=np.float32)

    slat = np.asarray(src_lat, dtype=np.float64)
    slon = np.asarray(src_lon, dtype=np.float64)
    if slat[0] > slat[-1]:
        src_data = np.flipud(src_data)
        slat = slat[::-1]

    # only valid pixels (finite and > 0) enter the log10 transform
    ok = np.isfinite(src_data) & (src_data > 0)
    logv = np.full_like(src_data, np.nan, dtype=np.float64)
    logv[ok] = np.log10(src_data[ok])

    TLAT, TLON = np.meshgrid(np.asarray(tgt_lat), np.asarray(tgt_lon), indexing='ij')
    try:
        interp = RegularGridInterpolator(
            (slat, slon), logv,
            bounds_error=False, fill_value=np.nan
        )
        pts = np.stack([TLAT.ravel(), TLON.ravel()], -1)
        out = interp(pts).reshape(len(tgt_lat), len(tgt_lon))
        out = np.power(10.0, out)   # back-transform to the linear domain
        return out.astype(np.float32)
    except Exception:
        return np.full((len(tgt_lat), len(tgt_lon)), np.nan, dtype=np.float32)


def load_nsoas_to_goci_grid(date_str, lat_min, lat_max, lon_min, lon_max, idx):
    """Load NSOAS Chl and resample it onto the GOCI 128x128 grid (north-up, row0=north)."""
    if date_str not in idx:
        return None, None
    try:
        with h5py.File(idx[date_str], 'r') as f:
            chl = f['Geophysical Data/CHL'][:].astype(np.float32)
            la = f['Navigation Data/lat'][:].squeeze()
            lo = f['Navigation Data/lon'][:].squeeze()
    except Exception:
        return None, None

    chl[chl <= -998] = np.nan  # FillValue -> NaN

    tgt_lat_asc = np.linspace(lat_min, lat_max, TARGET_SIZE)
    tgt_lon_asc = np.linspace(lon_min, lon_max, TARGET_SIZE)

    result_s2n = resample_to_target(chl, np.asarray(la), np.asarray(lo),
                                    tgt_lat_asc, tgt_lon_asc)
    result = np.flipud(result_s2n)
    valid = np.isfinite(result) & (result > 0)

    return result, valid


def calibrate_one_file(npz_path, nsoas_index, overwrite=False):
    """Calibrate a single .npz file in place."""
    base = os.path.basename(npz_path)

    # Check whether it has already been calibrated
    if not overwrite:
        try:
            check = np.load(npz_path, allow_pickle=True)
            if 'NSOAS_Chl_Value' in check.keys():
                return 'skip'
        except Exception:
            pass

    try:
        data = np.load(npz_path, allow_pickle=True)

        # GOCI-2 raw data
        chl_ori = data['chl_ori_value'].copy()        # (128,128) row0=north
        chl_flag = data['chl_ori_flag'].astype(np.uint32)

        # Step 1: build the GOCI-2 self-validity mask (independent of NSOAS)
        chl_flag_valid = ((chl_flag & CHL_INVALID) == 0)
        ac_ori_flag = data.get('ac_ori_flag')
        if ac_ori_flag is not None:
            ac_ori_flag = ac_ori_flag.astype(np.uint32)
            ac_valid = ((ac_ori_flag & AC_INVALID) == 0)
        else:
            ac_valid = np.ones_like(chl_flag_valid, dtype=bool)

        goci_valid = (ac_valid & chl_flag_valid &
                      np.isfinite(chl_ori) & (chl_ori > 0))

        goci_n = goci_valid.sum()
        if goci_n < 50:
            return 'fail_few_valid'

        # Metadata
        lat_min = float(data['meta_lat_min'])
        lat_max = float(data['meta_lat_max'])
        lon_min = float(data['meta_lon_min'])
        lon_max = float(data['meta_lon_max'])
        date_str = base.split('_')[1]

        # Step 2: load NSOAS and resample it onto the GOCI 128x128 grid
        nsoas_chl, nsoas_valid = load_nsoas_to_goci_grid(
            date_str, lat_min, lat_max, lon_min, lon_max, nsoas_index
        )
        if nsoas_chl is None:
            nsoas_valid = np.zeros_like(goci_valid, dtype=bool)
            nsoas_chl = np.full_like(chl_ori, np.nan, dtype=np.float32)

        # ============================================================
        # Step 3: V4 Chl>3 anchoring
        #   scale = median(NSOAS) / median(GOCI)  over GOCI>=3 ∩ NSOAS>=3
        #   Additional constraint: calibration runs only if the NSOAS
        #   overlap ratio in the high-value region is >= 70%
        # ============================================================
        goci_high = goci_valid & (chl_ori >= CHL_ANCHOR)
        nsoas_high_valid = nsoas_valid & np.isfinite(nsoas_chl) & (nsoas_chl >= CHL_ANCHOR)
        anchor = goci_high & nsoas_high_valid

        goci_high_n = goci_high.sum()
        overlap_ratio = anchor.sum() / max(goci_high_n, 1)

        if anchor.sum() >= 10 and overlap_ratio >= NSOAS_OVERLAP_MIN:
            goci_med = np.median(chl_ori[anchor])
            nsoas_med = np.median(nsoas_chl[anchor])
            if goci_med >= 0.01 and nsoas_med >= 0.01:
                scale = float(np.clip(nsoas_med / goci_med, 0.1, 10.0))
            else:
                scale = 1.0
        else:
            scale = 1.0

        # Step 4: sigmoid soft-transition calibration
        #   w = sigmoid((GOCI - 3.5) / 0.5)
        #   Chl_Calib = GOCI x (1 + w x (scale - 1))
        w = sigmoid((chl_ori - CHL_TRANSITION) / CHL_TRANSITION_W)

        chl_calib = np.zeros_like(chl_ori, dtype=np.float32)
        chl_calib[goci_valid] = chl_ori[goci_valid] * (1.0 + w[goci_valid] * (scale - 1.0))
        chl_calib[chl_calib > CHL_MAX_VALID] = CHL_MAX_VALID

        # Step 5: normalize the calibrated Chl
        chl_calib_norm = np.zeros_like(chl_calib, dtype=np.float32)
        norm_valid = goci_valid & (chl_calib > 0) & np.isfinite(chl_calib)
        if np.any(norm_valid):
            chl_calib_norm[norm_valid] = np.clip(
                (np.log10(chl_calib[norm_valid]) - LOG_MIN) / LOG_SPAN, 0.0, 1.0
            )

        # Step 6: update the .npz in place
        save_dict = {}
        for key in data.keys():
            save_dict[key] = data[key]

        delta_field = np.zeros_like(chl_ori, dtype=np.float32)
        delta_field[goci_valid] = np.log10(1.0 + w[goci_valid] * (scale - 1.0))
        block_valid = np.ones((1, 1), dtype=np.int8)

        save_dict['NSOAS_Chl_Value'] = nsoas_chl.astype(np.float32)
        save_dict['NSOAS_Valid_Mask'] = nsoas_valid.astype(np.int8)
        save_dict['NSOAS_Resolution'] = np.array("4km")
        save_dict['Chl_CalibByNSOAS'] = chl_calib.astype(np.float32)
        save_dict['Chl_Calib_Mask'] = goci_valid.astype(np.int8)
        save_dict['Calib_Delta_Field'] = delta_field.astype(np.float32)
        save_dict['Calib_Block_Valid'] = block_valid.astype(np.int8)
        save_dict['ChlCalib_Norm_Value'] = chl_calib_norm.astype(np.float32)

        np.savez_compressed(npz_path, **save_dict)
        return 'success'

    except Exception as e:
        return f'error:{str(e)[:80]}'


def main():
    parser = argparse.ArgumentParser(description='GOCI-2 Chl NSOAS calibration V4')
    parser.add_argument('--dir', required=True, help='Sample directory (e.g. H:\\250mOCDataset2)')
    parser.add_argument('--nsoas_root', default=NSOAS_ROOT,
                        help='NSOAS L3A product root directory (default: the NSOAS_ROOT variable)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite already-calibrated samples')
    parser.add_argument('--max', type=int, default=None, help='Maximum number of files to process (for testing)')
    args = parser.parse_args()

    npz_dir = args.dir
    if not os.path.exists(npz_dir):
        print(f"Directory does not exist: {npz_dir}")
        return

    nsoas_index = build_nsoas_index(args.nsoas_root)

    npz_files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
    if args.max:
        npz_files = npz_files[:args.max]

    print(f"\nProcessing directory: {npz_dir}")
    print(f"Total files: {len(npz_files)}")
    print(f"Scheme: V4 Chl>3 anchoring + sigmoid soft transition\n")

    stats = {'success': 0, 'skip': 0, 'fail_nsoas': 0, 'fail_few_valid': 0,
             'fail_no_delta': 0, 'error': 0}

    for fp in tqdm(npz_files, desc="Calibration progress"):
        result = calibrate_one_file(fp, nsoas_index, overwrite=args.overwrite)
        if result.startswith('fail_') or result.startswith('error:'):
            key = result.split(':')[0] if ':' in result else result
            stats[key] = stats.get(key, 0) + 1
        else:
            stats[result] = stats.get(result, 0) + 1

    print(f"\nCompletion stats:")
    print(f"  Calibrated successfully: {stats['success']}")
    print(f"  Skipped (already calibrated): {stats.get('skip', 0)}")
    print(f"  No NSOAS data: {stats.get('fail_nsoas', 0)}")
    print(f"  Too few valid pixels: {stats.get('fail_few_valid', 0)}")
    print(f"  No valid delta: {stats.get('fail_no_delta', 0)}")
    print(f"  Errors: {stats.get('error', 0)}")


if __name__ == "__main__":
    main()
