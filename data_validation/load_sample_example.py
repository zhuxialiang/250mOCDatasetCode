# -*- coding: utf-8 -*-
"""
Minimal loading example for the GOCI-2 pixel-level chlorophyll-a dataset
(companion script of Supplementary Note S2.18).

Usage:
    python load_sample_example.py <path-to-sample.npz>

Example:
    python load_sample_example.py RH_20210515_100000_0001.npz

The script lists the 24 fields of one sample (name, shape, dtype) and
demonstrates how to build the model input/labels used in the baseline
experiments of the manuscript (see the manuscript's Methods and Usage Notes).
"""
import sys
import numpy as np


def main(path):
    data = np.load(path, allow_pickle=True)

    print(f"NPZ: {path}")
    print(f"fields ({len(data.files)}):")
    for k in sorted(data.files):
        a = data[k]
        if a.ndim == 0:
            print(f"  {k:<22} scalar        {a.dtype}")
        else:
            print(f"  {k:<22} {str(a.shape):<14} {a.dtype}")

    # ---- Model input (12-band normalized Rrs) ----
    ac_in = data['ac_norm_data']               # (12,128,128) float32

    # ---- Chl-a regression labels ----
    chl_label = data['chl_norm_value']         # (128,128) float32, [0,1]
    chl_mask = data['chl_norm_valid']          # (128,128) int8

    # ---- NSOAS-corrected Chl (standard DL retrieval label) ----
    calib_chl = data['ChlCalib_Norm_Value']    # (128,128) float32
    calib_mask = data['Chl_Calib_Mask']        # (128,128) int8

    # ---- Metadata ----
    cls_name = str(data['meta_type'])
    lat = (float(data['meta_lat_min']), float(data['meta_lat_max']))
    lon = (float(data['meta_lon_min']), float(data['meta_lon_max']))
    valid_ratio = float(data['meta_valid_ratio'])

    print(f"\nmeta_type    : {cls_name}")
    print(f"lat range    : {lat}")
    print(f"lon range    : {lon}")
    print(f"valid_ratio  : {valid_ratio:.3f}")
    print(f"AC input     : {ac_in.shape}, {ac_in.dtype}")
    print(f"Chl label    : {chl_label.shape}, valid pixels "
          f"{int(chl_mask.sum())}/{chl_mask.size}")
    print(f"Calib label  : {calib_chl.shape}, mask pixels {int(calib_mask.sum())}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
