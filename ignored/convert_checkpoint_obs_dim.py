"""Utility to adapt an old RSL-RL checkpoint with a larger observation dimension
to a newer environment that uses a smaller observation vector.

Problem
-------
You modified the observation space (e.g. removed 4 features: old dim 38 -> new dim 34)
and now resuming training fails with:
  RuntimeError: size mismatch for actor.0.weight ... [256, 38] vs [256, 34]

Solution
--------
This script loads the checkpoint, slices (or reorders if you pass indices) the
first linear layer weights (actor.0.weight / critic.0.weight) and the running
statistics tensors (obs_rms / critic_obs_rms) to match the new dimension, then
saves a new compatible checkpoint.

Usage
-----
python scripts/convert_checkpoint_obs_dim.py \
  --in /path/to/old_checkpoint.pt \
  --out /path/to/converted_checkpoint.pt \
  --new-obs-dim 34

Optional: specify which columns (features) to keep if you removed something
from the middle instead of the end:
  --keep-cols 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
             24 25 26 27 28 29 30 31 32 33
If --keep-cols is omitted, the script assumes truncation: keep the first N.

Safety
------
The original file is not modified. Inspect the printed key summary to verify.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Slice observation dimension in RSL-RL checkpoint")
    p.add_argument("--in", dest="input_path", required=True, help="Path to original checkpoint (.pt)")
    p.add_argument("--out", dest="output_path", required=True, help="Path to write converted checkpoint")
    p.add_argument("--new-obs-dim", type=int, required=True, help="Target (smaller) observation dimension")
    p.add_argument(
        "--keep-cols",
        type=int,
        nargs="*",
        help="Explicit column indices to retain (length must equal new-obs-dim). If omitted, keep first N columns.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report intended changes without writing output file",
    )
    return p.parse_args()


LINEAR_WEIGHT_KEYS = ["actor.0.weight", "critic.0.weight"]
OBS_RMS_PREFIXES = ["obs_rms", "critic_obs_rms"]  # typical running mean/std containers


def slice_tensor(t: torch.Tensor, cols: list[int]) -> torch.Tensor:
    if t.dim() == 2:  # (out_features, in_features)
        return t[:, cols]
    elif t.dim() == 1:  # (features,)
        return t[cols]
    else:
        raise ValueError(f"Unexpected tensor rank {t.dim()} for slicing: shape={tuple(t.shape)}")


def main():
    args = parse_args()
    inp = Path(args.input_path)
    if not inp.is_file():
        print(f"ERROR: Input file not found: {inp}")
        sys.exit(1)

    ckpt = torch.load(str(inp), map_location="cpu")
    if "model_state_dict" not in ckpt:
        print("ERROR: 'model_state_dict' key not found in checkpoint. Not an RSL-RL checkpoint?")
        sys.exit(1)

    state = ckpt["model_state_dict"]
    # Determine old obs dim from one of the known weights
    old_obs_dim = None
    for k in LINEAR_WEIGHT_KEYS:
        if k in state:
            old_obs_dim = state[k].shape[1]
            break
    if old_obs_dim is None:
        print("ERROR: Could not infer old observation dimension (missing first layer weights).")
        sys.exit(1)

    new_obs_dim = args.new_obs_dim
    if new_obs_dim > old_obs_dim:
        print(f"ERROR: new_obs_dim ({new_obs_dim}) > old_obs_dim ({old_obs_dim}). Upsizing not supported.")
        sys.exit(1)

    if args.keep_cols:
        cols = args.keep_cols
        if len(cols) != new_obs_dim:
            print(
                f"ERROR: Provided {len(cols)} --keep-cols but --new-obs-dim is {new_obs_dim}. They must match."
            )
            sys.exit(1)
        if max(cols) >= old_obs_dim or min(cols) < 0:
            print("ERROR: One or more --keep-cols indices are out of range.")
            sys.exit(1)
    else:
        cols = list(range(new_obs_dim))  # assume truncation

    print("--- Checkpoint Observation Dimension Conversion ---")
    print(f"Input file      : {inp}")
    print(f"Old obs dim     : {old_obs_dim}")
    print(f"New obs dim     : {new_obs_dim}")
    print(f"Keeping columns : {cols}")

    # Adjust linear layer weights
    modified = []
    for key in LINEAR_WEIGHT_KEYS:
        if key in state:
            w = state[key]
            state[key] = slice_tensor(w, cols)
            modified.append(key)
    # Adjust running mean/std stats if present
    for k, v in list(state.items()):
        # Typical keys: 'obs_rms.mean', 'obs_rms.var', 'critic_obs_rms.mean', etc.
        for prefix in OBS_RMS_PREFIXES:
            if k.startswith(prefix) and v.dim() == 1 and v.shape[0] == old_obs_dim:
                state[k] = slice_tensor(v, cols)
                modified.append(k)

    print("Modified tensors:")
    for m in modified:
        print(f"  {m}: new shape {tuple(state[m].shape)}")

    if args.dry_run:
        print("Dry run: no file written.")
        return

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(out))
    print(f"Converted checkpoint written to: {out}")
    print("You can now resume training with the new environment (obs dim reduced).")


if __name__ == "__main__":  # pragma: no cover
    main()
