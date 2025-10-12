#!/usr/bin/env python3
"""Lightweight analyzer for an IsaacLab PPO training run directory.

Usage:
  python analyze_training_run.py --run logs/rsl_rl/so_arm100_picknplace_v1/2025-08-26_17-16-04 \
    [--tags reward success loss entropy value episode]

Reports per-tag latest value, mean(last N), slope(last N) and flags simple anomalies:
  - Plateau: abs(slope) < plateau_tol and history span > min_points
  - Collapse (entropy): latest < entropy_min
  - Divergence (loss): slope > loss_spike_tol (positive upward trend)
  - Success stagnation: tag contains 'success' and latest < success_floor after warmup_iters

Requires tensorboard event processing libs (tensorboard). Gracefully degrades if unavailable.
"""
from __future__ import annotations
import argparse
import math
import os
import statistics
from dataclasses import dataclass
from typing import List, Dict, Sequence

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
    TB_AVAILABLE = True
except Exception:  # pragma: no cover - tensorboard not installed
    TB_AVAILABLE = False

@dataclass
class TagStats:
    tag: str
    latest_step: int
    latest_value: float
    mean_last: float | None
    slope_last: float | None  # per step (simple linear regression)
    count: int
    anomalies: List[str]


def linear_regression_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def analyze_events(run_dir: str, tags_filter: List[str], last_points: int = 200) -> List[TagStats]:
    if not TB_AVAILABLE:
        raise RuntimeError("tensorboard not available; install with `pip install tensorboard`.")

    size_guidance = {  # keep memory bounded
        'scalars': 0,  # load all for selected tags; we'll slice later
    }
    acc = EventAccumulator(run_dir, size_guidance=size_guidance)
    acc.Reload()

    all_scalar_tags = acc.Tags().get('scalars', [])
    target_tags = [t for t in all_scalar_tags if any(f.lower() in t.lower() for f in tags_filter)]

    results: List[TagStats] = []
    for tag in sorted(target_tags):
        events = acc.Scalars(tag)
        if not events:
            continue
        # slice last points
        tail = events[-last_points:]
        steps = [e.step for e in tail]
        values = [float(e.value) for e in tail]
        latest_step = steps[-1]
        latest_value = values[-1]
        mean_last = statistics.fmean(values) if len(values) >= 2 else values[0]
        slope_last = linear_regression_slope(steps, values)

        anomalies: List[str] = []
        # Heuristics
        plateau_tol = 1e-4  # near-flat
        if slope_last is not None and abs(slope_last) < plateau_tol and len(values) > 50:
            anomalies.append('plateau')
        if 'entropy' in tag.lower() and latest_value < 0.01:
            anomalies.append('entropy_collapse')
        if 'loss' in tag.lower() and slope_last is not None and slope_last > 0.001:
            anomalies.append('loss_rising')
        if 'success' in tag.lower() and latest_step > 2000 and latest_value < 0.05:
            anomalies.append('success_stagnation')
        if 'reward' in tag.lower() and slope_last is not None and slope_last < -0.001:
            anomalies.append('reward_decline')

        results.append(TagStats(tag, latest_step, latest_value, mean_last, slope_last, len(events), anomalies))
    return results


def format_report(stats: List[TagStats]) -> str:
    lines = []
    header = f"{'TAG':50} {'STEP':>8} {'LATEST':>12} {'MEAN(last)':>12} {'SLOPE':>12} ANOMALIES"
    lines.append(header)
    lines.append('-' * len(header))
    for s in stats:
        slope_str = f"{s.slope_last:.3e}" if s.slope_last is not None else ' ' * 12
        mean_str = f"{s.mean_last:.4g}" if s.mean_last is not None else ' ' * 12
        anomalies = ','.join(s.anomalies)
        lines.append(f"{s.tag:50} {s.latest_step:8d} {s.latest_value:12.4g} {mean_str:12} {slope_str:12} {anomalies}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='Path to run directory containing events file')
    ap.add_argument('--tags', nargs='*', default=['reward', 'success', 'loss', 'entropy', 'value', 'episode'])
    ap.add_argument('--last-points', type=int, default=200)
    args = ap.parse_args()

    # Resolve events file parent if user passed params/ or events file directly
    run_dir = args.run
    if os.path.isfile(run_dir) and 'tfevents' in os.path.basename(run_dir):
        run_dir = os.path.dirname(run_dir)

    if not any('tfevents' in f for f in os.listdir(run_dir)):
        raise SystemExit(f'No events file found in {run_dir}')

    if not TB_AVAILABLE:
        raise SystemExit('tensorboard package not installed. Run: pip install tensorboard')

    stats = analyze_events(run_dir, args.tags, last_points=args.last_points)
    if not stats:
        print('No matching scalar tags found.')
        return
    print(format_report(stats))

    # Aggregate anomaly summary
    flagged = [s for s in stats if s.anomalies]
    if flagged:
        print('\nAnomalies detected:')
        for s in flagged:
            print(f' - {s.tag}: {";".join(s.anomalies)}')
    else:
        print('\nNo anomalies flagged by heuristics.')

if __name__ == '__main__':  # pragma: no cover
    main()
