#!/usr/bin/env python3
"""Plot a completed benchmark_amd.py summary; requires matplotlib."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
args = parser.parse_args()
status = json.loads((args.results/'status.json').read_text())
if not status or any(r['status'] != 'PASS' for r in status):
    raise SystemExit('Refusing to plot an incomplete or failed case set')
rows = list(csv.DictReader((args.results/'summary.csv').open()))
gm = json.loads((args.results/'geomean.json').read_text())['speedup']
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
for ax, stage, title in zip(axes, ['lshape', 'flt'], ['L-shape', 'FLT']):
    subset = [r for r in rows if r['stage'] == stage]
    values = [float(r['speedup']) for r in subset]+[gm[stage]]
    labels = [r['case'] for r in subset]+['Geo. mean']
    ax.bar(range(len(values)), values, color=['#416fa6']*(len(values)-1)+['#d28e35'])
    ax.axhline(1, color='0.4', linewidth=1, linestyle='--')
    ax.set_ylabel('Original / optimized')
    ax.set_title(f'{title}: batching function + complete DAG interval')
    ax.set_xticks(range(len(values)), labels, rotation=35, ha='right')
    ax.set_ylim(0, max(values)*1.2)
    for i, value in enumerate(values): ax.text(i, value, f'{value:.2f}×', ha='center', va='bottom', fontsize=8)
fig.savefig(args.results/'amd-speedup.pdf')
fig.savefig(args.results/'amd-speedup.png', dpi=180)
