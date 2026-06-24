# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# SPDX-License-Identifier: BSD-3-Clause
"""Batch-eval Lift Play checkpoints from one RSL-RL run directory.

Each checkpoint spawns a separate ``eval_lift_success_rate`` process (one Isaac Kit
startup per checkpoint). Results are written under ``--output_root`` and summarized
in ``summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def default_isaac_python() -> str:
    return os.environ.get("ISAAC_PYTHON", str(Path.home() / "isaacsim_v5.1" / "python.sh"))


def checkpoint_iter(path: Path) -> int:
    m = re.match(r"model_(\d+)\.pt$", path.name)
    if not m:
        msg = f"Not a model_<iter>.pt file: {path}"
        raise ValueError(msg)
    return int(m.group(1))


def discover_checkpoints(run_dir: Path) -> list[Path]:
    paths = list(run_dir.glob("model_*.pt"))
    if not paths:
        print(f"[ERROR] No model_*.pt under {run_dir}", file=sys.stderr)
        sys.exit(2)
    return sorted(paths, key=checkpoint_iter)


def select_checkpoints(paths: list[Path], *, every: int, iters: list[int] | None) -> list[Path]:
    if iters is not None:
        want = set(iters)
        sel = [p for p in paths if checkpoint_iter(p) in want]
        missing = sorted(want - {checkpoint_iter(p) for p in sel})
        if missing:
            print(f"[WARN] No checkpoint on disk for iterations: {missing}", file=sys.stderr)
        return sorted(sel, key=checkpoint_iter)
    if every > 0:
        last_i = checkpoint_iter(paths[-1])
        sel = [p for p in paths if checkpoint_iter(p) % every == 0 or checkpoint_iter(p) == last_i]
        seen: set[Path] = set()
        out: list[Path] = []
        for p in sorted(sel, key=checkpoint_iter):
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out
    return list(paths)


def _parse_result_line(log: str) -> tuple[int | None, int | None, float | None]:
    """Parse '[RESULT] episodes=N, success=M, rate=R%' from eval stdout."""
    for line in log.splitlines():
        if "[RESULT]" not in line:
            continue
        # episodes=130, success=23, rate=17.69%
        em = re.search(r"episodes=(\d+)", line)
        sm = re.search(r"success=(\d+)", line)
        rm = re.search(r"rate=([\d.]+)%", line)
        if em and sm and rm:
            return int(em.group(1)), int(sm.group(1)), float(rm.group(1))
    return None, None, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval_lift_success_rate on many model_*.pt files in one run dir.",
        epilog="Unknown flags are forwarded to eval_lift_success_rate (e.g. --headless --device cuda).",
    )
    parser.add_argument("--run_dir", type=Path, required=True, help="RSL-RL run folder containing model_*.pt")
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help=f"Isaac Sim python.sh (default: env ISAAC_PYTHON or {default_isaac_python()}).",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=None,
        help="cd here before launching eval (default: cwd).",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=500,
        help="Keep checkpoints where iteration %% every == 0, plus always the highest iteration on disk.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="If set, only these iteration indices (overrides --every).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Parent directory for per-checkpoint folders. Default: <run_dir>/batch_eval_<UTC time>.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if <output_subdir>/report.html already exists.",
    )
    parser.add_argument("--max_jobs", type=int, default=None, help="Evaluate at most this many checkpoints.")

    args, eval_extra = parser.parse_known_args()

    iters = args.iters
    if iters is not None and len(iters) == 0:
        iters = None

    banned = {"--checkpoint", "--output_dir"}
    if any(x in banned for x in eval_extra):
        print("[ERROR] Do not pass --checkpoint or --output_dir in extra args; the batch script sets them.", file=sys.stderr)
        sys.exit(2)

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"[ERROR] Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(2)

    py = args.python or default_isaac_python()
    if not args.dry_run and not os.path.isfile(py):
        print(f"[ERROR] Isaac python not found: {py}", file=sys.stderr)
        sys.exit(2)

    repo_root = (args.repo_root or Path.cwd()).resolve()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = (args.output_root or (run_dir / f"batch_eval_{ts}")).expanduser().resolve()

    paths = discover_checkpoints(run_dir)
    selected = select_checkpoints(paths, every=args.every, iters=iters)
    if args.max_jobs is not None:
        selected = selected[: max(0, args.max_jobs)]

    if not selected:
        print("[ERROR] No checkpoints selected after filters.", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] output_root={out_root}")
    print(f"[INFO] checkpoints={len(selected)}: {[checkpoint_iter(p) for p in selected]}")

    rows: list[dict[str, str | int | float]] = []

    for ckpt in selected:
        it = checkpoint_iter(ckpt)
        sub = out_root / f"iter_{it:06d}"
        report = sub / "report.html"
        if args.skip_existing and report.is_file():
            print(f"[SKIP] exists {report}")
            continue

        cmd = [
            py,
            "-m",
            "isaac_so_arm101.scripts.eval_lift_success_rate",
            "--checkpoint",
            str(ckpt),
            "--output_dir",
            str(sub),
        ]
        cmd.extend(eval_extra)

        print(f"\n[RUN] iter={it}\n  {' '.join(cmd)}")
        if args.dry_run:
            rows.append({"iteration": it, "checkpoint": str(ckpt), "episodes": "", "success": "", "rate_pct": "", "output_dir": str(sub), "status": "dry_run"})
            continue

        sub.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("ISAAC_PATH", str(Path.home() / "isaacsim_v5.1"))
        env.setdefault("CARB_APP_PATH", str(Path.home() / "isaacsim_v5.1" / "kit"))
        env.setdefault("EXP_PATH", str(Path.home() / "isaacsim_v5.1" / "apps"))
        env.pop("PYTHONPATH", None)

        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        log = proc.stdout + "\n" + proc.stderr
        ep, succ, rate = _parse_result_line(log)
        status = "ok" if proc.returncode == 0 and ep is not None else f"fail_rc={proc.returncode}"
        if proc.returncode != 0:
            print(log[-4000:], file=sys.stderr)
        rows.append(
            {
                "iteration": it,
                "checkpoint": str(ckpt),
                "episodes": ep if ep is not None else "",
                "success": succ if succ is not None else "",
                "rate_pct": rate if rate is not None else "",
                "output_dir": str(sub),
                "status": status,
            }
        )
        (sub / "eval_console.log").write_text(log, encoding="utf-8")
        print(f"[DONE] iter={it} status={status} episodes={ep} success={succ} rate={rate}")

    if args.dry_run:
        return

    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iteration", "checkpoint", "episodes", "success", "rate_pct", "output_dir", "status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[INFO] Wrote {csv_path}")


if __name__ == "__main__":
    main()
