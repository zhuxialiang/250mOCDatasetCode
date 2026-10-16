"""
Extract_Sample_Records.py - Extract metadata from .npz sample files and generate
a sample_records CSV.

Input:  all .npz files under the Train/Val/Test directories of H:\250mOCDataset\
Output: sample_records_final.csv in the current working directory

Columns:
  'id' (index), 'chl_file' (Chl file name), 'sample_type' (sample type),
  'bbox' (bounding box), 'valid_ratio' (valid pixel ratio)
"""

import os, glob, csv
import numpy as np

NPZ_ROOT = r"H:\250mOCDataset2"
OUT_CSV = "sample_records_final.csv"


def main():
    npz_files = []
    for sub in ['Train', 'Val', 'Test']:
        d = os.path.join(NPZ_ROOT, sub)
        if os.path.exists(d):
            npz_files.extend(glob.glob(os.path.join(d, '*.npz')))

    print(f'Found {len(npz_files)} .npz files')

    rows = []
    errors = 0

    for fp in npz_files:
        try:
            data = np.load(fp, allow_pickle=True)
            sample_id = int(data['meta_sample_id'])
            filename = str(data['meta_filename'])
            sample_type = str(data['meta_type'])
            lat_min = float(data['meta_lat_min'])
            lat_max = float(data['meta_lat_max'])
            lon_min = float(data['meta_lon_min'])
            lon_max = float(data['meta_lon_max'])
            valid_ratio = float(data['meta_valid_ratio'])

            area_str = f"({lat_min:.4f}, {lat_max:.4f}, {lon_min:.4f}, {lon_max:.4f})"
            rows.append((sample_id, filename, sample_type, area_str, valid_ratio))
            data.close()
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f'  Error: {os.path.basename(fp)} - {e}')

    # Sort records by sample_id
    rows.sort(key=lambda x: x[0])

    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'chl_file', 'sample_type', 'bbox', 'valid_ratio'])
        for r in rows:
            writer.writerow(r)

    print(f'\nDone: {len(rows)} records -> {OUT_CSV}')
    if errors:
        print(f'Errors: {errors}')

    # Statistics
    from collections import Counter
    types = Counter(r[2] for r in rows)
    print(f'\nType distribution:')
    for t, c in types.most_common():
        print(f'  {t}: {c}')


if __name__ == '__main__':
    main()
