"""
GOCI-2 East China Sea red tide sample data structure dictionary (updated 2026-07-16 V4)
Purpose: Define all fields, shapes, types, semantics and value rules of .npz sample files,
         so that downstream models can automatically identify and select the appropriate data.

File naming rule:
    {type_prefix}_{YYYYMMDD}_{HHMMSS}_{seq}.npz
    Type prefix: RH=red tide high-value zone, RB=red tide edge zone, PW=pure water, HZ=turbid water

Update history:
    2026-07-16: V4 calibration scheme — Chl>3 anchoring + Sigmoid soft transition
               Calib_Delta_Field changed to a pixel-level variable field (24 fields)
    2026-07-15: V2 calibration scheme — global median ratio (per-window independent scale)
               Chl_Calib_Mask changed to GOCI-2's own valid area, independent of NSOAS
               New field ChlCalib_Norm_Value added (24 fields)
    2026-05-30: NSOAS calibration fields added (7 fields), RH threshold updated 10→7,
               HZ Chl upper limit 4→2.5, chl_norm_valid logic for the four types updated
"""

import numpy as np

# ============================================================================
# 1. GOCI-2 12-band wavelength mapping
# ============================================================================
BAND_WAVELENGTH = {
    0:  ('Rrs_380', 380),
    1:  ('Rrs_412', 412),
    2:  ('Rrs_443', 443),
    3:  ('Rrs_490', 490),
    4:  ('Rrs_510', 510),
    5:  ('Rrs_555', 555),
    6:  ('Rrs_620', 620),
    7:  ('Rrs_660', 660),
    8:  ('Rrs_680', 680),
    9:  ('Rrs_709', 709),
    10: ('Rrs_745', 745),
    11: ('Rrs_865', 865),
}

BAND_NAMES = [BAND_WAVELENGTH[i][0] for i in range(12)]
BAND_WL = [BAND_WAVELENGTH[i][1] for i in range(12)]

# ============================================================================
# 2. AC flag bitmask definitions
# ============================================================================
AC_FLAG = {
    'cloud':        1,        # bit 0: cloud
    'land':         2,        # bit 1: land
    'ac_fail':      4,        # bit 2: atmospheric correction failed
    'extrm_turbid': 8,        # bit 3: extremely turbid water
    'sunglint':     16,       # bit 4: sunglint
    'saturation':   32,       # bit 5: sensor saturation
    'low_snr':      64,       # bit 6: low signal-to-noise ratio
    'turbid_water': 262144,   # bit 18: turbid water
}

AC_INVALID_MASK = (AC_FLAG['cloud'] | AC_FLAG['land'] | AC_FLAG['ac_fail'] |
                   AC_FLAG['sunglint'] | AC_FLAG['saturation'] | AC_FLAG['low_snr'])

AC_TURBID_MASK = AC_FLAG['extrm_turbid'] | AC_FLAG['turbid_water']

# ============================================================================
# 3. Chl flag bitmask definitions
# ============================================================================
CHL_FLAG = {
    'cloud':      1,      # bit 0: cloud
    'land':       2,      # bit 1: land
    'fail':       4,      # bit 2: inversion failed
    'sunglint':   16,     # bit 4: sunglint
    'saturation': 32,     # bit 5: sensor saturation
    'low_snr':    64,     # bit 6: low signal-to-noise ratio
}

CHL_INVALID_MASK = (CHL_FLAG['cloud'] | CHL_FLAG['land'] |
                    CHL_FLAG['sunglint'] | CHL_FLAG['saturation'] | CHL_FLAG['low_snr'])

# ============================================================================
# 4. .npz file internal data structure dictionary (24 fields)
# ============================================================================
SAMPLE_DATA_DICT = {
    # ---- 4.1 AC raw data ----
    'ac_ori_data': {
        'shape':        (12, 128, 128),
        'dtype':        np.float32,
        'description':  'AC raw Rrs data, 12 bands sorted by wavelength in ascending order',
        'bands':        BAND_NAMES,
        'wavelengths':  BAND_WL,
        'value_range':  'Raw values, typically -0.01 ~ 0.15 sr⁻¹',
        'invalid_pixels': 'NaN/Inf set to 0',
    },
    'ac_ori_flag': {
        'shape':        (128, 128),
        'dtype':        np.uint32,
        'description':  'AC flag bitmask; each bit marks a pixel state',
        'bit_definitions': AC_FLAG,
    },

    # ---- 4.2 AC normalized data ----
    'ac_norm_data': {
        'shape':        (12, 128, 128),
        'dtype':        np.float32,
        'description':  'AC normalized Rrs, formula: clip(Rrs, 0, 0.1) / 0.1',
        'value_range':  '[0, 1], invalid pixels = 0',
        'normalization': 'linear clipping normalization',
    },
    'ac_norm_valid': {
        'shape':        (128, 128),
        'dtype':        np.int8,
        'description':  'AC valid mask, computed from flags',
        'rule':         '~((ac_flag & AC_INVALID_MASK) != 0)',
        'values':       '0=invalid (cloud/land/sunglint/saturation/low SNR), 1=valid',
    },

    # ---- 4.3 Chl raw data ----
    'chl_ori_value': {
        'shape':        (128, 128),
        'dtype':        np.float32,
        'description':  'Chl raw chlorophyll concentration',
        'unit':         'mg/m³',
        'value_range':  '0 ~ 1000',
        'positive_samples': 'Real Chl values for RH/RB samples (red-tide high-value/boundary)',
        'negative_samples': 'Raw downloaded values (reference) for PW/HZ samples (pure/turbid water)',
    },
    'chl_ori_flag': {
        'shape':        (128, 128),
        'dtype':        np.uint32,
        'description':  'Chl flag bitmask',
        'bit_definitions': CHL_FLAG,
    },

    # ---- 4.4 Chl normalized data (training labels) ----
    'chl_norm_value': {
        'shape':        (128, 128),
        'dtype':        np.float32,
        'description':  'Chl normalized label value',
        'RH/RB_rule':   'norm = clip((log10(chl) - (-2.0)) / 4.0, 0, 1), Chl∈[0.01,100]→[0,1]',
        'PW/HZ_rule':   'Recovered real log-normalized values (Scheme B, see Chl_Norm.py); chl_norm_valid restricts the loss region',
        'value_range':  '[0, 1]',
    },
    'chl_norm_valid': {
        'shape':        (128, 128),
        'dtype':        np.int8,
        'description':  'Chl training valid mask (controls the loss computation region)',
        'RH_rule':      'chl_flag valid & Chl≥7, and high-value ratio ≥5% (denominator shrunk by raising low_cut)',
        'RB_rule':      '~((chl_flag & CHL_INVALID_MASK)!=0) & finite & chl>0 & chl<=1000',
        'PW_rule':      'ac_valid & (Chl<1) & ~ac_turbid & isfinite',
        'HZ_rule':      'ac_valid & (ac_flag & AC_TURBID_MASK != 0) & (Chl<2.5) & isfinite',
        'values':       '0=not involved in loss, 1=involved in loss',
    },

    # ---- 4.5 meta information ----
    'meta_sample_id': {
        'shape':        (),
        'dtype':        np.int32,
        'description':  'Sample ID, corresponding to the sequence in sample_records.csv',
    },
    'meta_filename': {
        'shape':        (),
        'dtype':        np.str_,
        'description':  'Original GOCI-II Chl file name',
    },
    'meta_type': {
        'shape':        (),
        'dtype':        np.str_,
        'description':  'Sample type: RH/RB/PW/HZ (red-tide high/boundary, pure water, turbid water)',
    },
    'meta_valid_ratio': {
        'shape':        (),
        'dtype':        np.float32,
        'description':  'Ratio of valid Chl pixels within the window (valid Chl pixels / total pixels)',
    },
    'meta_lat_min': {
        'shape':        (),
        'dtype':        np.float32,
        'description':  'Minimum latitude of the window (°N)',
    },
    'meta_lat_max': {
        'shape':        (),
        'dtype':        np.float32,
        'description':  'Maximum latitude of the window (°N)',
    },
    'meta_lon_min': {
        'shape':        (),
        'dtype':        np.float32,
        'description':  'Minimum longitude of the window (°E)',
    },
    'meta_lon_max': {
        'shape':        (),
        'dtype':        np.float32,
        'description':  'Maximum longitude of the window (°E)',
    },

    # ========================================================================
    # 4.6 NSOAS calibration data (added 2026-05-30)
    # ========================================================================
    'NSOAS_Chl_Value': {
        'shape':        (128, 128),
        'dtype':        np.float32,
        'description':  'NSOAS 4km Chl resampled to the GOCI grid via bilinear interpolation (250m for GOCI-2, 500m for GOCI-1)',
        'unit':         'mg/m³',
        'source':       'NSOAS multi-source merged L3A daily product',
        'resampling':   'RegularGridInterpolator bilinear',
    },
    'NSOAS_Valid_Mask': {
        'shape':        (128, 128),
        'dtype':        np.int8,
        'description':  'NSOAS valid pixel mask (not NaN, Chl>0)',
        'values':       '0=invalid, 1=valid',
    },
    'NSOAS_Resolution': {
        'shape':        (),
        'dtype':        np.str_,
        'description':  'Native spatial resolution of the NSOAS product',
        'value':        '4km',
    },
    'Chl_CalibByNSOAS': {
        'shape':        (128, 128),
        'dtype':        np.float32,
        'description':  'Chl calibrated with NSOAS Chl>3 anchoring + Sigmoid soft transition (invalid area=0)',
        'unit':         'mg/m³',
        'method':       'V4: anchor=GOCI≥3∩NSOAS≥3, scale=clip(median(NSOAS)/median(GOCI),0.1,10.0), w=sigmoid((GOCI-3.5)/0.5), Chl_Calib=GOCI×(1+w×(scale-1))',
        'calib_mask':   'Only valid where Chl_Calib_Mask=1',
        'hard_cap':     '≤ 100 mg/m³ (upper limit of the GOCI-2 OC algorithm calibration domain)',
    },
    'Chl_Calib_Mask': {
        'shape':        (128, 128),
        'dtype':        np.int8,
        'description':  'Calibration valid region mask (V4: GOCI-2 own valid area, independent of NSOAS)',
        'rule':         'GOCI AC_valid ∩ Chl not cloud/land ∩ Chl>0',
        'values':       '0=not calibrated, 1=calibrated',
    },
    'Calib_Delta_Field': {
        'shape':        (128, 128),
        'dtype':        np.float32,
        'description':  'Bias field in log domain (V4: pixel-level variable field)',
        'formula':      'Δ = log10(1 + w × (scale-1)), w=sigmoid((GOCI-3.5)/0.5)',
        'value_range':  '≈0 for low-Chl pixels, ≈log10(scale) for high-Chl pixels',
    },
    'Calib_Block_Valid': {
        'shape':        ('n×n', 'variable'),
        'dtype':        np.int8,
        'description':  'Always 1×1 under V4, marks whether calibration is valid (always 1)',
        'values':       '1=valid (V4 per-window calibration has no block logic)',
    },

    # ========================================================================
    # 4.7 Calibrated Chl normalization (added 2026-07)
    # ========================================================================
    'ChlCalib_Norm_Value': {
        'shape':        (128, 128),
        'dtype':        np.float32,
        'description':  'Log-normalized value of calibrated Chl (V4 Sigmoid soft transition)',
        'formula':      'norm = clip((log10(Chl_Calib) - (-2.0)) / 4.0, 0, 1)',
        'value_range':  '[0, 1]',
        'valid_region': 'Valid only where Chl_Calib_Mask=1 and Chl>0; 0 elsewhere',
        'usage':        'Standard training/validation/test label for deep-learning Chl-a retrieval',
    },
}

# ============================================================================
# 5. Sample types and label strategy (updated 2026-05-30)
# ============================================================================
SAMPLE_TYPE_CONFIG = {
    'RH': {
        'prefix':          'RH',
        'is_positive':     True,
        'chl_label_mode':  'real',
        'chl_threshold':   7,          # Chl ≥ 7 mg/m³
        'window_density':  0.01,        # coverage threshold of the 128×128 window
        'description':     'Red tide bloom core zone, Chl≥7, used for Chl regression. chl_norm_valid covers only Chl≥7 areas with high-value ratio ≥5%',
    },
    'RB': {
        'prefix':          'RB',
        'is_positive':     True,
        'chl_label_mode':  'real',
        'chl_high':        7,          # high-value threshold
        'chl_low':         7,          # low-value threshold (unified to 7 with the high-value zone)
        'window_density':  0.01,
        'description':     'Transition zone between red tide and normal water, mixed high/low Chl, all valid Chl pixels in the window participate in regression',
    },
    'PW': {
        'prefix':          'PW',
        'is_positive':     False,
        'chl_label_mode':  'zero',
        'chl_max':         1,
        'window_density':  0.90,
        'description':     'Oligotrophic water, Chl<1, non-turbid, chl_norm_valid covers only (Chl<1 & ~turbid) areas',
    },
    'HZ': {
        'prefix':          'HZ',
        'is_positive':     False,
        'chl_label_mode':  'zero',
        'chl_max':         2.5,        # Chl < 2.5 (excludes red-tide confusion)
        'window_density':  0.10,
        'description':     'High suspended-sediment area, AC flags turbid + Chl<2.5, chl_norm_valid covers only turbid areas',
    },
}

# ============================================================================
# 6. Data loading and validation utility functions
# ============================================================================
def load_sample(filepath):
    """Load a single sample file and verify structural completeness"""
    data = np.load(filepath, allow_pickle=True)
    required = list(SAMPLE_DATA_DICT.keys())
    # Ignore legacy fields that have been removed from the dictionary
    for key in required:
        if key not in data:
            # NSOAS fields may be uncalibrated; not mandatory
            if key.startswith('NSOAS') or key.startswith('Calib') or key.startswith('Chl_Calib'):
                continue
            raise KeyError(f"Missing required field: {key}")
    return data


def parse_filename(filepath):
    """Parse sample meta information from the file name

    Returns:
        dict: {prefix, date, time, seq}
        Example: RH_20210429_061530_17.npz -> {prefix: 'RH', date: '20210429', time: '061530', seq: 17}
    """
    import os, re
    base = os.path.basename(filepath)
    m = re.match(r'([A-Z]+)_(\d{8})_(\d{6})_(\d+)\.npz', base)
    if not m:
        raise ValueError(f"File name format does not match: {base}")
    return {'prefix': m.group(1), 'date': m.group(2), 'time': m.group(3), 'seq': int(m.group(4))}


def get_input_tensor(data, band_indices=None):
    """Get the model input tensor

    Args:
        data:          dict returned by np.load
        band_indices:  list of band indices; None means all 12 bands
    Returns:
        (C, 128, 128) float32 normalized tensor
    """
    ac = data['ac_norm_data']
    if band_indices is not None:
        ac = ac[band_indices]
    return ac.astype(np.float32)


def get_target_tensor(data):
    """Get the Chl target tensor

    Returns:
        (1, 128, 128) float32 normalized Chl
    """
    return data['chl_norm_value'][np.newaxis].astype(np.float32)


def get_valid_mask(data, domain='ac'):
    """Get the valid pixel mask

    Args:
        domain: 'ac' | 'chl'
    Returns:
        (128, 128) bool
    """
    return data[f'{domain}_norm_valid'].astype(bool)


def get_meta(data):
    """Extract meta information as a dict"""
    return {
        'sample_id':   int(data['meta_sample_id']),
        'filename':    str(data['meta_filename']),
        'type':        str(data['meta_type']),
        'valid_ratio': float(data['meta_valid_ratio']),
        'lat_range':   (float(data['meta_lat_min']), float(data['meta_lat_max'])),
        'lon_range':   (float(data['meta_lon_min']), float(data['meta_lon_max'])),
    }


def get_flag(data, domain='ac'):
    """Get the raw flag data

    Returns:
        (128, 128) uint32
    """
    return data[f'{domain}_ori_flag']


def get_nsoas_calib(data):
    """Get NSOAS calibration data (if present)

    Returns:
        dict or None
    """
    if 'Chl_CalibByNSOAS' not in data:
        return None
    return {
        'chl_calib':      data['Chl_CalibByNSOAS'],
        'calib_mask':     data['Chl_Calib_Mask'].astype(bool),
        'delta_field':    data['Calib_Delta_Field'],
        'block_valid':    data['Calib_Block_Valid'],
        'nsoas_chl':      data['NSOAS_Chl_Value'],
        'nsoas_valid':    data['NSOAS_Valid_Mask'].astype(bool),
        'nsoas_res':      str(data['NSOAS_Resolution']),
    }


if __name__ == "__main__":
    print("GOCI-2 sample data structure dictionary (updated 2026-07-16 V4)")
    print(f"AC bands: {BAND_NAMES}")
    print(f"AC flags: {AC_FLAG}")
    print(f"Chl flags: {CHL_FLAG}")
    print(f"Number of fields: {len(SAMPLE_DATA_DICT)} (including 7 NSOAS calibration fields)")
    print(f"Sample types: {list(SAMPLE_TYPE_CONFIG.keys())}")
    print()
    print("Key changes:")
    print("  - RH threshold: Chl≥10 → Chl≥7")
    print("  - HZ Chl upper limit: <4 → <2.5")
    print("  - Negative-sample chl_norm_valid: changed from =ac_valid to area-specific mask")
    print("  - RH chl_norm_valid: only Chl≥7 areas, high-value ratio ≥5%")
    print("  - Added 7 NSOAS calibration fields (group 4.6)")
