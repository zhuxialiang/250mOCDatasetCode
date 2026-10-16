# GOCI-2 Pixel-Level Chlorophyll-a Dataset: Processing & Validation Code

This repository contains the open-source processing and validation code accompanying the dataset paper:

> **A 250-m ocean color dataset with pixel-level chlorophyll-a annotations from GOCI-2 (2021–2024)** (Scientific Data, under review)

It covers the complete workflow used to build a machine-learning-ready dataset of **13,034 samples** (128×128-pixel, 32 km × 32 km windows, NPZ format, 24 fields per sample) over the East China Sea (24–34°N, 117–128°E) from GOCI-2 geostationary ocean color observations, including:

- sliding-window mining of four water-type samples (red-tide high **RH**, red-tide boundary **RB**, pure water **PW**, turbid water **HZ**);
- cropping and normalization of GOCI-2 Level-2 products;
- **V4 bias correction** of GOCI-2 chlorophyll-a (Chl-*a*) against the NSOAS multi-source merged product (Chl > 3 mg m⁻³ anchoring with a sigmoid soft transition);
- benchmark deep-learning validation (classification + regression) with IOCCG-standard metrics.

## Repository structure

```
SciData-code/
├── README.md
├── LICENSE                       # MIT License
├── requirements.txt
├── S2/                       # Sample-record CSVs at each pipeline stage
│   ├── RedEvent2021-2024.csv        # Red-tide event inventory (input to Step 1)
│   ├── sample_records.csv           # Step 1 output: candidate windows (108,938)
│   ├── sample_records_balanced.csv  # Stage-1 balanced crop list (18,706 windows)
│   └── sample_records_final.csv     # Final 13,034-sample manifest
├── data_processing/          # Dataset construction pipeline
│   ├── Find_4Class_Sample.py        # Step 1: sliding-window mining & four-class labeling
│   ├── Balance_Samples.py           # Stage-1 class balancing of the candidate list
│   ├── Sample_Crop.py               # Step 2: crop windows from GOCI-2 L2 NetCDF and normalize
│   ├── Chl_CalibByNSOAS.py          # Step 3: V4 bias correction against NSOAS (Chl>3 anchoring + sigmoid)
│   ├── Chl_Norm.py                   # Step 4 (optional): recover PW/HZ normalized labels from zero
│   ├── Extract_Sample_Records.py    # Step 5: generate the sample manifest CSV from final .npz files
│   └── sample_data_dictionary.py    # Data dictionary of the 24 NPZ fields, band mapping, flag definitions
└── data_validation/          # Dataset usability validation
    ├── load_sample_example.py    # Minimal NPZ loading example (Supplementary Note S2.18)
    ├── quick_test.py             # Data-logic self tests (no PyTorch required)
    ├── train_baseline.py         # U-Net baseline (groups A/B of paper Table 4)
    ├── eval_same_label.py        # Same-label evaluation (B/same-label row of Table 4)
    └── sensitivity_eval.py       # Evaluate existing weights on the log-domain release dataset
```

## Requirements

**Data processing** (Python ≥ 3.8):

```
numpy pandas xarray scipy tqdm h5py
```

**Data validation** (additional):

```
torch
```

Or simply: `pip install -r requirements.txt`.

## Input data sources

| Source | Product | Resolution | Purpose |
|---|---|---|---|
| KOSC (Korea Ocean Satellite Center) | GOCI-2 Level-2 AC (12-band Rrs) and Chl | 250 m | Source reflectance and Chl |
| NSOAS (National Satellite Ocean Application Service) | Multi-source merged L3A Chl | 4 km | Reference for bias correction |
| Official bulletins | Red tide event records (2021–2024, 154 events) | — | Spatial constraints for RH/RB/PW/HZ labeling |

*Note: the authors do not hold the copyright of the original input data; users must obtain their own authorization.*

## Processing pipeline

### Step 1 — Sample mining & four-class labeling

```bash
python data_processing/Find_4Class_Sample.py --data_dirs K:\GOCI2_Data\2021 K:\GOCI2_Data\2022
```

> **Path configuration**: pass your GOCI-2 L2 directories with `--data_dirs` (the red-tide event CSV defaults to `S2/RedEvent2021-2024.csv`, the output to `S2/sample_records.csv`). Edit the defaults in `__main__` if you prefer fixed paths.

Scans GOCI-2 AC/Chl file pairs with a sliding 128×128-pixel window (stride 64, 50% overlap) and classifies each window by Chl thresholds, AC turbidity flags, and red-tide event region constraints:

| Type | Chl condition | AC condition | Coverage | Spatial constraint |
|---|---|---|---|---|
| RH | Chl ≥ 7 mg m⁻³ | — | ≥ 1% | Window center in event region |
| RB | Chl ≥ 7 and < 7 coexist | — | both ≥ 1% | Event region + 0.05° buffer |
| PW | Chl < 1 mg m⁻³ | non-turbid | ≥ 90% | Outside event region + 0.2° |
| HZ | Chl < 2.5 mg m⁻³ | AC turbid | ≥ 10% | Outside event region + 0.2° |

Output: a sample record CSV (columns: id, chl_file, sample_type, bbox, valid_ratio) listing candidate windows. Class balance is adjusted in two stages (see the paper's Methods for details): stage 1 subsamples the redundant PW/HZ classes (monthly quota + valid-pixel ratio priority, keeping all RH/RB) to produce the 18,706-window crop list; stage 2 applies equidistant (time-balanced) subsampling to the dominant PW class after cropping. The subsequent pixel-count quality filtering and NSOAS-calibration cleanup are applied on the cropped samples, yielding the final 13,034-window list. Run `python data_processing/Balance_Samples.py --help` for the reference implementation of the reproducible stages.

### Step 2 — Cropping & normalization

```bash
python data_processing/Sample_Crop.py --csv <records.csv> --out <output_dir>
```

> **Path configuration**: edit the `DATA_DIRS` list at the top of `Sample_Crop.py` to your GOCI-2 L2 directories before running.

Extracts each window from the GOCI-2 L2 NetCDF files and saves a 128×128-pixel NPZ sample. Normalization:

- AC Rrs: `ac_norm = clip(Rrs, 0, 0.1) / 0.1`
- Chl: `chl_norm = clip((log10(Chl) + 2.0) / 4.0, 0, 1)` (Chl ∈ [0.01, 100] mg m⁻³ → [0, 1])

### Step 3 — V4 bias correction

```bash
python data_processing/Chl_CalibByNSOAS.py --dir <sample_dir> --nsoas_root <NSOAS_root> [--overwrite]
```

> **Path configuration**: pass your NSOAS L3A download root with `--nsoas_root` (the author's local path in `NSOAS_ROOT` is only a fallback).

Anchors the GOCI-2 Chl bias in the high-Chl regime with the NSOAS 4 km product (bilinearly resampled to the GOCI grid):

- anchor region: GOCI ≥ 3 ∩ NSOAS ≥ 3 mg m⁻³ (requires NSOAS overlap ≥ 70%)
- `scale = clip(median(NSOAS) / median(GOCI), 0.1, 10.0)`
- weight `w = sigmoid((GOCI − 3.5) / 0.5)`, `Chl_calib = GOCI × (1 + w × (scale − 1))`, hard-capped at 100 mg m⁻³
- low-Chl pixels (w ≈ 0) retain the original GOCI values; high-Chl pixels (w ≈ 1) are fully anchored; the sigmoid ensures a smooth transition

### Step 4 (optional) — PW/HZ label recovery

```bash
python data_processing/Chl_Norm.py <sample_dir> [--dry-run]
```

Restores the real log-normalized Chl labels of PW/HZ negative samples.

### Step 5 — Sample manifest

```bash
python data_processing/Extract_Sample_Records.py
```

Generates the manifest CSV (`sample_records_final.csv`) listing all final samples with water-type labels and geographic bounds, enabling fast spatial/temporal queries and reproduction of the statistics reported in the paper.

### Data dictionary

```bash
python data_processing/sample_data_dictionary.py
```

Prints the complete definitions of the 24 NPZ fields (shape, dtype, semantics, value rules), the GOCI-2 12-band wavelength mapping, the AC/Chl flag bit definitions, and the four-class configuration.

## Validation

### Data-logic self tests (no GPU/PyTorch required)

```bash
python data_validation/quick_test.py
```

### Baseline benchmarks

`train_baseline.py` trains the U-Net baseline with two label schemes (original GOCI labels, group A; V4-corrected labels, group B) and evaluates both against the independent NSOAS reference; `eval_same_label.py` reproduces the same-label row of Table 4:

```bash
python data_validation/train_baseline.py --model unet --label ori --epochs 100
python data_validation/train_baseline.py --model unet --label calib --epochs 100
python data_validation/eval_same_label.py --model unet --label calib --ckpt data_validation/weights/unet_calib.pth
```

Metrics follow the IOCCG/NASA SeaBASS validation protocol (linear- and log-space RMSE/MAE/Bias/R², MDPD, SI, threshold-based F1).

## Dataset

The dataset itself is deposited at Zenodo:

> Zhu, X. (2026). *A 250-m ocean color dataset with pixel-level chlorophyll-a annotations from GOCI-2 (2021–2024)* [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.21831012>

Released under **CC BY 4.0**. Users must cite both this paper and the dataset DOI.

## License

This code is released under the **MIT License**.

## Citation

If you use this code, please cite:

> Zhu, X. (2026). A 250-m ocean color dataset with pixel-level chlorophyll-*a* annotations from GOCI-2 (2021–2024). *Scientific Data*. [DOI to be assigned]
