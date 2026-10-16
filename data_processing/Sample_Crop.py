"""
Sample_Crop.py — Crop 128x128 pixel windows from GOCI-2 Level-2 NetCDF files
according to a sample-record CSV (columns: 'sample_type', 'chl_file', 'bbox',
'valid_ratio', 'id'), normalize the patches, and save them as .npz files.

Default sample-record file: ..\\S2\\sample_records_final.csv (relative to this script)
Output structure: <out>/Train, <out>/Val, <out>/Test, assigned by the year-based
split of the paper (2021+2023 -> Train, 2022 -> Val, 2024 -> Test).

Normalization:
  - AC Rrs: clip(Rrs, 0, 0.1) / 0.1
  - Chl:    clip((log10(Chl) + 2.0) / 4.0, 0, 1), i.e. Chl in [0.01, 100] -> [0, 1]

Sample types: 'RH' (red tide high-value zone), 'RB' (red tide edge zone),
'PW' (pure water), 'HZ' (turbid water).

Usage:
  python Sample_Crop.py                                  # use the default CSV/output
  python Sample_Crop.py --csv <path> --out <dir>
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import logging
from datetime import datetime

TARGET_SIZE = 128

BANDS_12 = ['Rrs_380', 'Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_510', 'Rrs_555',
            'Rrs_620', 'Rrs_660', 'Rrs_680', 'Rrs_709', 'Rrs_745', 'Rrs_865']

AC_MASK_CLOUD = 1
AC_MASK_LAND = 2
AC_MASK_AC_FAIL = 4
AC_MASK_SUNGLINT = 16
AC_MASK_SATURATION = 32
AC_MASK_LOW_SNR = 64
AC_MASK_INVALID = (AC_MASK_CLOUD | AC_MASK_LAND | AC_MASK_AC_FAIL |
                   AC_MASK_SUNGLINT | AC_MASK_SATURATION | AC_MASK_LOW_SNR)

CHL_MASK_CLOUD = 1
CHL_MASK_LAND = 2
CHL_MASK_SUNGLINT = 16
CHL_MASK_SATURATION = 32
CHL_MASK_LOW_SNR = 64
CHL_MASK_INVALID = (CHL_MASK_CLOUD | CHL_MASK_LAND |
                    CHL_MASK_SUNGLINT | CHL_MASK_SATURATION | CHL_MASK_LOW_SNR)

AC_MASK_TURBID = 8 | 262144
TURBID_CHL_MAX = 2.5
HAB_HIGH_CHL_MIN = 7

TYPE_PREFIX = {
    'RH': 'RH',
    'RB': 'RB',
    'PW': 'PW',
    'HZ': 'HZ'
}

DATA_DIRS = [
    r"K:\GOCI2_Data\2021",
    r"K:\GOCI2_Data\2022",
    r"J:\GOCI2_Data\2023",
    r"I:\GOCI2_Data\2024"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_logging(output_root):
    """Set up a logger with both file and console handlers under output_root."""
    os.makedirs(output_root, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_root, 'sample_crop.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('Sample_Crop')


def build_file_index(data_dirs):
    """Build a {filename: full_path} index of all .nc files under the given directories."""
    logger = logging.getLogger('Sample_Crop')
    index = {}
    for d in data_dirs:
        if not os.path.exists(d):
            logger.warning(f"Directory does not exist: {d}")
            continue
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.endswith('.nc'):
                    full = os.path.join(root, f)
                    if f not in index:
                        index[f] = full
    return index


def parse_area(area_str):
    """Parse an area string into [lat_min, lat_max, lon_min, lon_max]; return None on failure."""
    try:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(area_str))
        if len(nums) == 4:
            return [float(n) for n in nums]
        return None
    except Exception:
        return None


def get_pixel_slice(lats, lons, bounds):
    """Return (y_slice, x_slice) covering all pixels within the given lat/lon bounds."""
    lat_min, lat_max, lon_min, lon_max = bounds
    mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
    indices = np.argwhere(mask)
    if indices.size == 0:
        return None, None
    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)
    return slice(y_min, y_max + 1), slice(x_min, x_max + 1)


def align_to_128_2d(data):
    """Pad/crop a 2D array to (TARGET_SIZE, TARGET_SIZE) with zeros."""
    h, w = data.shape
    result = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=data.dtype)
    h_copy = min(h, TARGET_SIZE)
    w_copy = min(w, TARGET_SIZE)
    result[:h_copy, :w_copy] = data[:h_copy, :w_copy]
    return result


def align_to_128_3d(data):
    """Pad/crop a 3D array (channels, h, w) to (channels, TARGET_SIZE, TARGET_SIZE) with zeros."""
    c, h, w = data.shape
    result = np.zeros((c, TARGET_SIZE, TARGET_SIZE), dtype=data.dtype)
    h_copy = min(h, TARGET_SIZE)
    w_copy = min(w, TARGET_SIZE)
    result[:, :h_copy, :w_copy] = data[:, :h_copy, :w_copy]
    return result


def infer_ac_filename(chl_filename):
    """Derive the corresponding AC NetCDF filename from a Chl filename."""
    return chl_filename.replace('_Chl.nc', '_AC.nc').replace('_chl.nc', '_ac.nc')


def parse_date_time_from_filename(filename):
    """Extract (date, time) strings from a GOCI-2 L2 filename; return (None, None) if unmatched."""
    match = re.search(r'_L2_(\d{8})_(\d{6})_', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def year_to_split(year):
    """Map a scene year to the train/validation/test split (paper's year-based split)."""
    if year in ('2021', '2023'):
        return 'Train'
    if year == '2022':
        return 'Val'
    if year == '2024':
        return 'Test'
    return None  # unexpected year


def load_ac_data(ac_path, y_slice, x_slice):
    """Load Rrs bands, AC flag and navigation lat/lon for the given pixel slice."""
    with xr.open_dataset(ac_path, group='geophysical_data/Rrs') as ds_rrs:
        rrs_stack = []
        for band in BANDS_12:
            if band in ds_rrs:
                band_data = ds_rrs[band].isel(
                    number_of_lines=y_slice, pixels_per_line=x_slice
                ).values
            else:
                band_data = np.full(
                    (y_slice.stop - y_slice.start, x_slice.stop - x_slice.start),
                    np.nan, dtype=np.float32
                )
            rrs_stack.append(band_data)
        ac_data = np.stack(rrs_stack, axis=0).astype(np.float32)

    with xr.open_dataset(ac_path, group='geophysical_data') as ds_flag:
        ac_flag = ds_flag['flag'].isel(
            number_of_lines=y_slice, pixels_per_line=x_slice
        ).values.astype(np.uint32)

    with xr.open_dataset(ac_path, group='navigation_data') as ds_nav:
        lats = ds_nav['latitude'].isel(
            number_of_lines=y_slice, pixels_per_line=x_slice
        ).values
        lons = ds_nav['longitude'].isel(
            number_of_lines=y_slice, pixels_per_line=x_slice
        ).values

    return ac_data, ac_flag, lats, lons


def load_chl_data(chl_path, y_slice, x_slice):
    """Load the Chl variable and flag for the given pixel slice."""
    with xr.open_dataset(chl_path, group='geophysical_data') as ds:
        chl_var = 'Chl' if 'Chl' in ds.variables else 'chlor_a'
        chl_value = ds[chl_var].isel(
            number_of_lines=y_slice, pixels_per_line=x_slice
        ).values.astype(np.float32)

        chl_flag = np.zeros_like(chl_value, dtype=np.uint32)
        if 'flag' in ds.variables:
            chl_flag = ds['flag'].isel(
                number_of_lines=y_slice, pixels_per_line=x_slice
            ).values.astype(np.uint32)

    return chl_value, chl_flag


def compute_ac_norm(ac_data, ac_flag):
    """Normalize AC Rrs with clip(Rrs, 0, 0.1)/0.1; return (ac_norm, ac_valid)."""
    ac_invalid = (ac_flag & AC_MASK_INVALID) != 0
    ac_valid = ~ac_invalid

    ac_norm = np.zeros_like(ac_data, dtype=np.float32)
    for i in range(ac_data.shape[0]):
        band = ac_data[i].copy()
        nan_mask = np.isnan(band) | np.isinf(band)
        band[nan_mask] = 0.0
        ac_norm[i] = np.clip(band, 0.0, 0.1) / 0.1

    ac_norm[:, ac_invalid] = 0.0

    return ac_norm, ac_valid


def compute_chl_norm_positive(chl_value, chl_flag):
    """Log-normalize positive Chl with clip((log10(Chl)+2)/4, 0, 1); return (chl_norm, chl_valid)."""
    chl_invalid = (chl_flag & CHL_MASK_INVALID) != 0
    chl_valid = ~chl_invalid & np.isfinite(chl_value) & (chl_value > 0) & (chl_value <= 1000)

    log_chl = np.zeros_like(chl_value, dtype=np.float32)
    log_chl[chl_valid] = np.log10(chl_value[chl_valid])
    chl_norm = np.zeros_like(chl_value, dtype=np.float32)
    chl_norm[chl_valid] = np.clip((log_chl[chl_valid] - (-2.0)) / 4.0, 0.0, 1.0)

    return chl_norm, chl_valid


def process_sample_row(row, file_index, output_root, logger):
    """Crop one sample patch from the CSV row and save it as a .npz file; return a status string."""

    chl_filename = row['chl_file']
    sample_type = row['sample_type']
    area_str = row['bbox']
    valid_ratio = row['valid_ratio']
    sample_id = row['id']

    year, time_str = parse_date_time_from_filename(chl_filename)
    if year is None:
        logger.warning(f"Failed to parse date: {chl_filename}")
        return 'fail'

    split_dir = year_to_split(year)
    if split_dir is None:
        logger.warning(f"Unexpected year {year}, skipped: {chl_filename}")
        return 'fail'
    output_dir = os.path.join(output_root, split_dir)
    os.makedirs(output_dir, exist_ok=True)

    prefix = TYPE_PREFIX.get(sample_type, 'UN')
    output_name = f"{prefix}_{year}_{time_str}_{sample_id}.npz"
    output_path = os.path.join(output_dir, output_name)

    if os.path.exists(output_path):
        return 'skip'

    if chl_filename not in file_index:
        logger.warning(f"File not found: {chl_filename}")
        return 'fail'
    chl_path = file_index[chl_filename]

    ac_filename = infer_ac_filename(chl_filename)
    if ac_filename not in file_index:
        logger.warning(f"AC file not found: {ac_filename}")
        return 'fail'
    ac_path = file_index[ac_filename]

    bounds = parse_area(area_str)
    if bounds is None:
        logger.warning(f"Area parsing failed: {area_str}")
        return 'fail'

    try:
        with xr.open_dataset(chl_path, group='navigation_data') as ds_nav:
            lats_full = ds_nav['latitude'].values
            lons_full = ds_nav['longitude'].values

        y_slice, x_slice = get_pixel_slice(lats_full, lons_full, bounds)
        if y_slice is None:
            logger.warning(f"Unable to locate in the scene: {chl_filename} {area_str}")
            return 'fail'

        ac_data_raw, ac_flag_raw, ac_lats, ac_lons = load_ac_data(ac_path, y_slice, x_slice)
        chl_value_raw, chl_flag_raw = load_chl_data(chl_path, y_slice, x_slice)

        ac_data = align_to_128_3d(ac_data_raw)
        ac_flag = align_to_128_2d(ac_flag_raw)
        chl_value = align_to_128_2d(chl_value_raw)
        chl_flag = align_to_128_2d(chl_flag_raw)
        patch_lats = align_to_128_2d(ac_lats)
        patch_lons = align_to_128_2d(ac_lons)

        ac_norm, ac_valid = compute_ac_norm(ac_data, ac_flag)

        if sample_type in ('PW', 'HZ'):
            chl_norm = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
            if sample_type == 'HZ':
                sample_area = ((ac_flag & AC_MASK_TURBID) != 0) & (chl_value < TURBID_CHL_MAX) & (chl_value >= 0) & np.isfinite(chl_value)
            else:
                ac_is_turbid = ((ac_flag & AC_MASK_TURBID) != 0)
                sample_area = (chl_value < 1) & (chl_value >= 0) & np.isfinite(chl_value) & ~ac_is_turbid
            chl_valid = ac_valid & sample_area
        elif sample_type == 'RH':
            chl_norm, chl_valid_full = compute_chl_norm_positive(chl_value, chl_flag)
            base_valid = chl_valid_full
            total_base = int(np.sum(base_valid))
            total_high = int(np.sum(base_valid & (chl_value >= HAB_HIGH_CHL_MIN)))
            if total_base > 0 and total_high > 0:
                ratio = total_high / total_base
                low_cut = 1.0
                while ratio < 0.05 and low_cut < HAB_HIGH_CHL_MIN:
                    low_cut += 0.5
                    base_valid = chl_valid_full & (chl_value >= low_cut)
                    total_base = int(np.sum(base_valid))
                    total_high = int(np.sum(base_valid & (chl_value >= HAB_HIGH_CHL_MIN)))
                    if total_base > 0:
                        ratio = total_high / total_base
                    else:
                        break
            chl_valid = base_valid & (chl_value >= HAB_HIGH_CHL_MIN)
        else:
            chl_norm, chl_valid = compute_chl_norm_positive(chl_value, chl_flag)

        save_dict = {
            'ac_ori_data': ac_data.astype(np.float32),
            'ac_ori_flag': ac_flag.astype(np.uint32),
            'ac_norm_data': ac_norm.astype(np.float32),
            'ac_norm_valid': ac_valid.astype(np.int8),
            'chl_ori_value': chl_value.astype(np.float32),
            'chl_ori_flag': chl_flag.astype(np.uint32),
            'chl_norm_value': chl_norm.astype(np.float32),
            'chl_norm_valid': chl_valid.astype(np.int8),
            'meta_sample_id': np.array(sample_id, dtype=np.int32),
            'meta_filename': np.array(chl_filename),
            'meta_type': np.array(sample_type),
            'meta_valid_ratio': np.array(valid_ratio, dtype=np.float32),
            'meta_lat_min': np.array(patch_lats.min(), dtype=np.float32),
            'meta_lat_max': np.array(patch_lats.max(), dtype=np.float32),
            'meta_lon_min': np.array(patch_lons.min(), dtype=np.float32),
            'meta_lon_max': np.array(patch_lons.max(), dtype=np.float32),
        }

        np.savez_compressed(output_path, **save_dict)
        return 'success'

    except Exception as e:
        logger.error(f"Processing failed [{chl_filename}]: {e}")
        return 'fail'


def main():
    """Command-line entry point: build the file index, read the CSV, and crop all samples."""
    parser = argparse.ArgumentParser(description='GOCI-2 Sample Crop Tool')
    parser.add_argument('--csv', default=r'..\S2\sample_records_final.csv',
                        help='Sample-record CSV path, relative to this script (default: ..\\S2\\sample_records_final.csv)')
    parser.add_argument('--out', default=r"H:\250mOCDataset2",
                        help='Output root with Train/Val/Test subfolders (default: H:\\250mOCDataset2)')
    args = parser.parse_args()

    # Resolve the CSV path against the script location so the script works from any CWD
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.normpath(os.path.join(SCRIPT_DIR, csv_path))
    output_root = args.out

    os.makedirs(output_root, exist_ok=True)
    logger = setup_logging(output_root)
    logger.info(f"CSV: {csv_path}, output root: {output_root}")

    logger.info("Building file index...")
    file_index = build_file_index(DATA_DIRS)
    logger.info(f"File index built, {len(file_index)} files in total")

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    logger.info(f"CSV loaded, {len(df)} records in total")

    type_counts = df['sample_type'].value_counts().to_dict()
    for k, v in type_counts.items():
        logger.info(f"  {k}: {v}")

    stats = {'success': 0, 'skip': 0, 'fail': 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Crop progress"):
        status = process_sample_row(row, file_index, output_root, logger)
        stats[status] += 1

    logger.info(f"Processing complete: success={stats['success']}, skipped (already exists)={stats['skip']}, failed={stats['fail']}")
    logger.info(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()
