"""
Find_4Class_Sample.py - Mine 128x128-pixel (32 km x 32 km) sample windows from
GOCI-2 Level-2 products over the East China Sea (24-34N, 117-128E) and classify
them into four sample types for the GOCI-2 chlorophyll dataset described in the
companion Scientific Data paper.

Input:
    - GOCI-2 Level-2 Chl and AC (atmospheric correction) NetCDF files, searched
      recursively under the directories listed in data_dirs
    - A red tide event CSV (red_tide_events.csv) providing event date ranges and
      geographic bounding regions

Output:
    - A CSV file recording each mined sample window with the columns:
      'id' (index), 'chl_file' (Chl file name), 'sample_type' (sample type),
      'bbox' (bounding box), 'valid_ratio' (valid pixel ratio)

Algorithm (identical to the paper's methodology):
    - Slide a 128x128-pixel window over each scene with a step of 64 pixels,
      i.e., 50% overlap between neighbouring windows.
    - Classify each window into one of four types (values of the 'sample_type'
      column):
        1. 'RH' (Red Tide High-value): Chl >= 7 with a high-value
           pixel density >= 1% inside the window, constrained to red tide event
           regions.
        2. 'RB' (Red Tide Edge): coexistence of high (Chl >= 7) and
           low (0 <= Chl < 7) values, each with a density >= 1%, constrained to
           red tide event regions with a 0.05-degree buffer.
        3. 'PW' (Pure Water): Chl < 1 and non-turbid AC flag with a
           window coverage >= 90%, required to lie in the safe zone (outside red
           tide event regions, 0.2-degree buffer).
        4. 'HZ' (Turbid Water): AC turbid flag with Chl < 2.5 and a
           window coverage >= 10%, required to lie in the safe zone.

Usage:
    python Find_4Class_Sample.py
"""

import os
import re
import csv
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import logging


class RedTideEventParser:
    """Parse red tide event records from a CSV file (UTF-8 or GBK encoded)."""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.events = []
        self._parse_events()

    def _parse_events(self):
        """Read all valid events from the CSV into self.events."""
        encoding = self._detect_encoding(self.csv_path)
        with open(self.csv_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                event = self._parse_single_event(row)
                if event:
                    self.events.append(event)

    @staticmethod
    def _detect_encoding(csv_path: str) -> str:
        """Return the CSV encoding: UTF-8 (with or without BOM) if decodable, otherwise GBK."""
        with open(csv_path, 'rb') as f:
            raw = f.read(1024 * 64)
        for enc in ('utf-8-sig', 'utf-8', 'gbk'):
            try:
                raw.decode(enc)
                return enc
            except UnicodeDecodeError:
                continue
        return 'gbk'

    def _parse_single_event(self, row: Dict) -> Optional[Dict]:
        """Parse one CSV row into an event dict; return None for invalid rows."""
        try:
            date_str = row['BEDate']
            year_str = row['EYear']
            
            if not date_str or not year_str:
                return None

            dates = self._parse_date_range(date_str, int(year_str))
            if not dates:
                return None

            region = self._parse_region(row['Region'])
            if not region:
                return None

            return {
                'event_id': row['EventID'],
                'start_date': dates[0],
                'end_date': dates[1],
                'species': row['Species'],
                'lat_min': region[0],
                'lat_max': region[1],
                'lon_min': region[2],
                'lon_max': region[3]
            }
        except Exception as e:
            return None

    def _parse_date_range(self, date_str: str, year: int) -> Optional[Tuple[datetime, datetime]]:
        """Extract (start, end) datetimes from a date-range string (e.g. '4/22-5/7')."""
        try:
            match = re.search(r'(\d{1,2})[\u6708\/](\d{1,2})[\u65e5]?\s*[-~\u81f3]\s*(\d{1,2})[\u6708\/](\d{1,2})[\u65e5]?', date_str)
            if match:
                month1, day1, month2, day2 = match.groups()
                start_date = datetime(year, int(month1), int(day1))
                end_date = datetime(year, int(month2), int(day2))
                return start_date, end_date
        except:
            pass
        return None

    def _infer_year_from_context(self, date_str: str) -> Optional[int]:
        """Infer the year from an explicit 'YYYY\u5e74' pattern if present."""
        year_match = re.search(r'(\d{4})\u5e74', date_str)
        if year_match:
            return int(year_match.group(1))
        return None

    def _parse_region(self, region_str: str) -> Optional[Tuple[float, float, float, float]]:
        """Parse (lat_min, lat_max, lon_min, lon_max) from a region string with N/S/E/W degree markers."""
        try:
            coords = re.findall(r'(\d+\.\d+)°[NS]', region_str)
            if len(coords) >= 2:
                lat_coords = sorted([float(c) for c in coords[:2]])
            else:
                return None

            coords = re.findall(r'(\d+\.\d+)°[EW]', region_str)
            if len(coords) >= 2:
                lon_coords = sorted([float(c) for c in coords[:2]])
            else:
                return None

            return (lat_coords[0], lat_coords[1], lon_coords[0], lon_coords[1])
        except:
            return None

    def get_active_events(self, date: datetime) -> List[Dict]:
        """Return events active on the given date (inclusive of start/end)."""
        return [e for e in self.events if e['start_date'] <= date <= e['end_date']]

    def is_in_event_region(self, lat: float, lon: float, date: datetime, buffer_deg: float = 0.0) -> bool:
        """Return True if the point lies inside any active event region (plus buffer)."""
        active_events = self.get_active_events(date)
        for event in active_events:
            if (event['lat_min'] - buffer_deg <= lat <= event['lat_max'] + buffer_deg and
                event['lon_min'] - buffer_deg <= lon <= event['lon_max'] + buffer_deg):
                return True
        return False

    def is_in_safe_zone(self, lat: float, lon: float, date: datetime, buffer_deg: float = 0.2) -> bool:
        """Return True if the point lies outside every active event region (plus buffer)."""
        active_events = self.get_active_events(date)
        for event in active_events:
            if (event['lat_min'] - buffer_deg <= lat <= event['lat_max'] + buffer_deg and
                event['lon_min'] - buffer_deg <= lon <= event['lon_max'] + buffer_deg):
                return False
        return True


class SampleDetector:
    """Detect the four sample classes within a window using sliding density filters."""
    def __init__(self, window_size: int = 128, step_size: int = 64):
        self.window_size = window_size
        self.step_size = step_size

    def detect_hab_high(self, chl: np.ndarray, chl_flag: np.ndarray) -> np.ndarray:
        """RH: valid pixels with Chl >= 7 whose window density reaches 1%."""
        chl_valid = ~((chl_flag & (1 | 2 | 4)) != 0)
        hab_mask = chl_valid & (chl >= 7) & np.isfinite(chl)
        density = uniform_filter(hab_mask.astype(float), size=self.window_size, mode='constant', cval=0)
        return density >= 0.01

    def detect_hab_edge(self, chl: np.ndarray, chl_flag: np.ndarray) -> np.ndarray:
        """RB: coexistence of high (Chl >= 7) and low (0 <= Chl < 7) densities, both >= 1%."""
        chl_valid = ~((chl_flag & (1 | 2 | 4)) != 0)

        high_mask = chl_valid & (chl >= 7) & np.isfinite(chl)
        low_mask = chl_valid & (chl < 7) & (chl >= 0) & np.isfinite(chl)

        high_density = uniform_filter(high_mask.astype(float), size=self.window_size, mode='constant', cval=0)
        low_density = uniform_filter(low_mask.astype(float), size=self.window_size, mode='constant', cval=0)

        return (high_density >= 0.01) & (low_density >= 0.01)

    def detect_pure_water(self, chl: np.ndarray, chl_flag: np.ndarray, ac_flag: np.ndarray) -> np.ndarray:
        """PW: valid, non-turbid pixels with 0 <= Chl < 1, window coverage >= 90%."""
        chl_valid = ~((chl_flag & (1 | 2 | 4)) != 0)
        ac_valid = ~((ac_flag & (1 | 2 | 4)) != 0)
        ac_is_turbid = ((ac_flag & (8 | 262144)) != 0)

        pure_mask = chl_valid & ac_valid & ~ac_is_turbid & (chl < 1) & (chl >= 0) & np.isfinite(chl)
        density = uniform_filter(pure_mask.astype(float), size=self.window_size, mode='constant', cval=0)
        return density >= 0.9

    def detect_turbid_water(self, chl: np.ndarray, chl_flag: np.ndarray, ac_flag: np.ndarray) -> np.ndarray:
        """HZ: AC turbid-flagged pixels with 0 <= Chl < 2.5, window coverage >= 10%."""
        chl_valid = ~((chl_flag & (1 | 2 | 4)) != 0)
        ac_valid = ~((ac_flag & (1 | 2 | 4)) != 0)
        ac_is_turbid = ((ac_flag & (8 | 262144)) != 0)

        turbid_mask = chl_valid & ac_valid & ac_is_turbid & (chl < 2.5) & (chl >= 0) & np.isfinite(chl)
        density = uniform_filter(turbid_mask.astype(float), size=self.window_size, mode='constant', cval=0)
        return density >= 0.10


class SampleFinder:
    """Slide windows over GOCI-2 Chl/AC scenes, classify them into four types, and write the sample records CSV."""
    def __init__(self, data_dirs: List[str], red_tide_csv: str, output_csv: str, processed_filter: str = None):
        self.data_dirs = data_dirs
        self.red_tide_csv = red_tide_csv
        self.output_csv = output_csv
        self.event_parser = RedTideEventParser(red_tide_csv)
        self.detector = SampleDetector(window_size=128, step_size=64)
        self.processed_files = self._load_processed_files(filter_type=processed_filter)
        self.sample_counter = 0

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(os.path.dirname(output_csv), 'sample_finder.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_processed_files(self, filter_type: str = None) -> set:
        """Load Chl file names already recorded in the output CSV, optionally filtered by sample type."""
        processed = set()
        if os.path.exists(self.output_csv):
            try:
                df = pd.read_csv(self.output_csv, encoding='utf-8-sig')
                if filter_type:
                    processed = set(df[df['sample_type'] == filter_type]['chl_file'].tolist())
                else:
                    processed = set(df['chl_file'].tolist())
            except:
                pass
        return processed

    def _load_existing_rows(self):
        """Load previously recorded CSV rows so that processing can resume."""
        if os.path.exists(self.output_csv):
            try:
                df = pd.read_csv(self.output_csv, encoding='utf-8-sig')
                return df.to_dict('records')
            except:
                pass
        return []

    def _find_chl_ac_pairs(self) -> List[Tuple[str, str]]:
        """Recursively find (Chl, AC) file pairs that have not been processed yet."""
        pairs = []
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                continue

            for root, dirs, files in os.walk(data_dir):
                chl_files = [f for f in files if f.lower().endswith('_chl.nc')]
                for chl_file in chl_files:
                    chl_path = os.path.join(root, chl_file)
                    if chl_file in self.processed_files:
                        continue

                    ac_file = chl_file.replace('_chl.nc', '_ac.nc').replace('_Chl.nc', '_AC.nc')
                    ac_path = os.path.join(root, ac_file)

                    if os.path.exists(ac_path):
                        pairs.append((chl_path, ac_path))

        return pairs

    def _parse_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract the acquisition datetime from a filename with 'YYYYMMDD' and an optional '_HHMMSS_' part."""
        try:
            match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
            if match:
                year, month, day = match.groups()
                time_match = re.search(r'_(\d{6})_', filename)
                if time_match:
                    hour, minute, second = time_match.group(1)[:2], time_match.group(1)[2:4], time_match.group(1)[4:6]
                    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                else:
                    return datetime(int(year), int(month), int(day))
        except:
            pass
        return None

    def _load_data(self, chl_path: str, ac_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                              Optional[np.ndarray], Optional[np.ndarray],
                                                              Optional[np.ndarray], Optional[np.ndarray]]:
        """Load Chl, Chl flag, AC flag, latitude and longitude arrays from the NetCDF groups."""
        try:
            with xr.open_dataset(chl_path, group='geophysical_data') as ds_chl:
                chl = ds_chl['Chl'].values
                chl_flag = ds_chl['flag'].values.astype(np.uint32)

            with xr.open_dataset(ac_path, group='geophysical_data') as ds_ac:
                ac_flag = ds_ac['flag'].values.astype(np.uint32)

            with xr.open_dataset(chl_path, group='navigation_data') as ds_nav:
                lats = ds_nav['latitude'].values
                lons = ds_nav['longitude'].values

            return chl, chl_flag, ac_flag, lats, lons, None
        except Exception as e:
            return None, None, None, None, None, None

    def _analyze_window(self, chl: np.ndarray, chl_flag: np.ndarray, ac_flag: np.ndarray,
                       lats: np.ndarray, lons: np.ndarray, 
                       row_start: int, col_start: int,
                       img_date: datetime) -> List[Dict]:
        """Classify a single 128x128 window into zero or more sample dicts."""
        window_size = self.detector.window_size
        row_end = row_start + window_size
        col_end = col_start + window_size

        if row_end > chl.shape[0] or col_end > chl.shape[1]:
            return []

        chl_window = chl[row_start:row_end, col_start:col_end]
        chl_flag_window = chl_flag[row_start:row_end, col_start:col_end]
        ac_flag_window = ac_flag[row_start:row_end, col_start:col_end]
        lats_window = lats[row_start:row_end, col_start:col_end]
        lons_window = lons[row_start:row_end, col_start:col_end]

        lat_center = np.mean(lats_window)
        lon_center = np.mean(lons_window)

        samples = []

        is_hab_high = self.detector.detect_hab_high(chl_window, chl_flag_window)

        valid_mask = ~((chl_flag_window & (1 | 2 | 4)) != 0)
        valid_ratio = np.sum(valid_mask) / (window_size * window_size)

        # 1. Red Tide High-value zone (RH)
        is_hab_high = self.detector.detect_hab_high(chl_window, chl_flag_window)
        if is_hab_high.any():
            if self.event_parser.is_in_event_region(lat_center, lon_center, img_date):
                samples.append({
                    'sample_type': 'RH',
                    'lat_min': float(lats_window.min()),
                    'lat_max': float(lats_window.max()),
                    'lon_min': float(lons_window.min()),
                    'lon_max': float(lons_window.max()),
                    'valid_ratio': float(valid_ratio)
                })

        # 2. Red Tide Edge zone (RB)
        is_hab_edge = self.detector.detect_hab_edge(chl_window, chl_flag_window)
        if is_hab_edge.any():
            if self.event_parser.is_in_event_region(lat_center, lon_center, img_date, buffer_deg=0.05):
                samples.append({
                    'sample_type': 'RB',
                    'lat_min': float(lats_window.min()),
                    'lat_max': float(lats_window.max()),
                    'lon_min': float(lons_window.min()),
                    'lon_max': float(lons_window.max()),
                    'valid_ratio': float(valid_ratio)
                })

        # 3. Pure Water zone (PW)
        is_pure = self.detector.detect_pure_water(chl_window, chl_flag_window, ac_flag_window)
        if is_pure.any():
            if self.event_parser.is_in_safe_zone(lat_center, lon_center, img_date):
                samples.append({
                    'sample_type': 'PW',
                    'lat_min': float(lats_window.min()),
                    'lat_max': float(lats_window.max()),
                    'lon_min': float(lons_window.min()),
                    'lon_max': float(lons_window.max()),
                    'valid_ratio': float(valid_ratio)
                })

        # 4. Turbid Water zone (HZ)
        is_turbid = self.detector.detect_turbid_water(chl_window, chl_flag_window, ac_flag_window)
        if is_turbid.any():
            if self.event_parser.is_in_safe_zone(lat_center, lon_center, img_date):
                samples.append({
                    'sample_type': 'HZ',
                    'lat_min': float(lats_window.min()),
                    'lat_max': float(lats_window.max()),
                    'lon_min': float(lons_window.min()),
                    'lon_max': float(lons_window.max()),
                    'valid_ratio': float(valid_ratio)
                })

        return samples

    def _process_single_file(self, chl_path: str, ac_path: str) -> List[Dict]:
        """Mine all windows of one Chl/AC file pair and return sample record dicts."""
        chl_filename = os.path.basename(chl_path)
        img_date = self._parse_date_from_filename(chl_filename)

        if not img_date:
            return []

        chl, chl_flag, ac_flag, lats, lons, _ = self._load_data(chl_path, ac_path)
        if chl is None:
            return []

        samples = []
        window_size = self.detector.window_size
        step_size = self.detector.step_size

        for row_start in range(0, chl.shape[0] - window_size + 1, step_size):
            for col_start in range(0, chl.shape[1] - window_size + 1, step_size):
                window_samples = self._analyze_window(
                    chl, chl_flag, ac_flag, lats, lons,
                    row_start, col_start, img_date
                )
                for sample in window_samples:
                    self.sample_counter += 1
                    samples.append({
                        'id': self.sample_counter,
                        'chl_file': chl_filename,
                        'sample_type': sample['sample_type'],
                        'bbox': f"({sample['lat_min']:.4f}, {sample['lat_max']:.4f}, {sample['lon_min']:.4f}, {sample['lon_max']:.4f})",
                        'valid_ratio': round(sample['valid_ratio'], 4)
                    })

        return samples

    def _diagnose_turbid(self, filename, chl, chl_flag, ac_flag):
        """Log turbidity diagnostics for a scene (used for threshold tuning)."""
        chl_valid = ~((chl_flag & (1 | 2 | 4)) != 0)
        ac_valid = ~((ac_flag & (1 | 2 | 4)) != 0)
        ac_is_turbid = ((ac_flag & (8 | 262144)) != 0)

        total_pixels = ac_flag.size
        valid_pixels = int(np.sum(ac_valid))
        turbid_pixels_raw = int(np.sum(ac_is_turbid))
        turbid_valid_pixels = int(np.sum(ac_valid & ac_is_turbid))
        turbid_chl_under4_pixels = int(np.sum(ac_valid & ac_is_turbid & chl_valid & (chl < 4) & np.isfinite(chl) & (chl >= 0)))

        turbid_mask = chl_valid & ac_valid & ac_is_turbid & (chl < 4) & (chl >= 0) & np.isfinite(chl)
        turbid_density = uniform_filter(turbid_mask.astype(float), size=128, mode='constant', cval=0)
        max_turbid_density = float(np.max(turbid_density))

        self.logger.info(
            f"[Turbidity diagnostics] {filename} | "
            f"total_pixels={total_pixels} | "
            f"valid_pixels={valid_pixels} | "
            f"AC_turbid_raw={turbid_pixels_raw} | "
            f"AC_turbid_valid={turbid_valid_pixels} | "
            f"turbid_Chl_lt4_pixels={turbid_chl_under4_pixels} | "
            f"max_turbid_density_128win={max_turbid_density:.4f}"
        )

    def run(self, max_files: int = None):
        """Mine samples from all found file pairs and write (or append to) the results CSV."""
        self.logger.info(f"Starting sample mining, red tide events: {len(self.event_parser.events)}")

        pairs = self._find_chl_ac_pairs()
        self.logger.info(f"Found {len(pairs)} Chl+AC file pairs to process")

        if max_files:
            pairs = pairs[:max_files]
            self.logger.info(f"Test mode: processing only the first {max_files} pairs")

        existing_rows = self._load_existing_rows()
        max_existing_id = max([r['id'] for r in existing_rows]) if existing_rows else 0
        self.sample_counter = max_existing_id

        for chl_path, ac_path in tqdm(pairs, desc="Processing progress"):
            try:
                samples = self._process_single_file(chl_path, ac_path)
                if samples:
                    existing_rows.extend(samples)
                    self.logger.info(f"Processed: {os.path.basename(chl_path)}, samples found: {len(samples)}")
            except Exception as e:
                self.logger.error(f"Failed to process: {os.path.basename(chl_path)}, error: {e}")

        if existing_rows:
            df = pd.DataFrame(existing_rows)
            df.to_csv(self.output_csv, index=False, encoding='utf-8-sig')
            new_count = len(existing_rows) - (max_existing_id if max_existing_id else len(existing_rows))
            self.logger.info(f"Done! Total {len(existing_rows)} samples, results saved to: {self.output_csv}")
        else:
            self.logger.warning("No samples found")


if __name__ == "__main__":
    import argparse
    HERE = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='GOCI-2 four-class sample mining')
    parser.add_argument('--data_dirs', nargs='+', default=[
        r"K:\GOCI2_Data\2021",
        r"K:\GOCI2_Data\2022",
        r"J:\GOCI2_Data\2023",
        r"I:\GOCI2_Data\2024"
    ], help='GOCI-2 L2 NetCDF directories (modify to your download locations)')
    parser.add_argument('--events', default=os.path.normpath(os.path.join(HERE, '..', 'S2', 'RedEvent2021-2024.csv')),
                        help='Red tide event CSV (default: Code/S2/RedEvent2021-2024.csv)')
    parser.add_argument('--output', default=os.path.normpath(os.path.join(HERE, '..', 'S2', 'sample_records.csv')),
                        help='Output sample-record CSV (default: Code/S2/sample_records.csv)')
    parser.add_argument('--max_files', type=int, default=None, help='Test mode: only process the first N file pairs')
    args = parser.parse_args()

    finder = SampleFinder(args.data_dirs, args.events, args.output)
    finder.run(max_files=args.max_files)
