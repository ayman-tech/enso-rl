"""Migrate schema-v4 inference archives to centered event timing without simulation.

Usage: python scripts/analysis/center_inference.py plots/m1/inference.npz
The original archive is retained as inference.trailing-v4.npz. Interior centered
values equal the next stored trailing value; the last point averages the two
available months, matching running_mean's boundary convention.
"""
import argparse
from pathlib import Path
import sys
import os
import shutil
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.enso_classifier import classify_enso_event

LABELS = {'Neutral': 0, 'Single-year El Nino': 1, 'Single-year La Nina': 2,
          'Multi-year El Nino': 3, 'Multi-year La Nina': 4}

def migrate(path):
    path = Path(path)
    with np.load(path, allow_pickle=True) as archive:
        if int(archive['schema_version']) == 5:
            print(f'Already centered: {path}')
            return
        if int(archive['schema_version']) != 4 or int(archive['nino34_smooth_months']) != 3:
            raise ValueError('Migration requires a schema-v4 three-month archive')
        data = {key: archive[key] for key in archive.files}
    for system in ('agent', 'base'):
        previous = data[f'{system}_nino34_oni']
        centered = np.empty_like(previous)
        centered[..., :-1] = previous[..., 1:]
        centered[..., -1] = data[f'{system}_obs'][..., -2:, 0].mean(axis=-1)
        labels = np.empty_like(data[f'{system}_mye_label'])
        for index in np.ndindex(centered.shape[:-1]):
            # Recover the initial state's centered value from the old partial
            # two-point trailing mean, then classify before dropping that state.
            full = np.concatenate(([previous[index][0]], centered[index]))
            classes = classify_enso_event(full)
            labels[index] = [LABELS[c] for c in classes[1:]]
        data[f'{system}_nino34_oni'] = centered
        data[f'{system}_mye_label'] = labels
    data['schema_version'] = np.array(5)
    data['nino34_smooth_alignment'] = np.array('centered')
    data['event_timing_migrated_from'] = np.array('schema-v4 trailing mean')
    backup = path.with_name(path.stem + '.trailing-v4.npz')
    if backup.exists():
        raise FileExistsError(f'Backup already exists: {backup}')
    shutil.copy2(path, backup)
    temporary = path.with_name(path.stem + '.centered-tmp.npz')
    np.savez_compressed(temporary, **data)
    os.replace(temporary, path)
    print(f'Centered: {path}; original: {backup}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    for path in parser.parse_args().archives:
        migrate(path)
