#!/usr/bin/env bash
# Long Lift training (defaults: max_iterations=20000, num_envs=3072 from env/agent cfg),
# then evaluate the latest checkpoint (default 128 episodes; set EVAL_MAX_EPISODES) and write HTML/PNG under logs/eval_after_long_train/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

unset PYTHONPATH
export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_v5.1}"
export CARB_APP_PATH="${CARB_APP_PATH:-$HOME/isaacsim_v5.1/kit}"
export EXP_PATH="${EXP_PATH:-$HOME/isaacsim_v5.1/apps}"

PY="${ISAAC_PYTHON:-$HOME/isaacsim_v5.1/python.sh}"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: Isaac Sim python not found at $PY (set ISAAC_PYTHON to override)" >&2
  exit 1
fi

echo "[$(date -Iseconds)] Starting Lift long train (SO-ARM101)…"
"$PY" -m isaac_so_arm101.scripts.rsl_rl.train \
  --task Isaac-SO-ARM101-Lift-Cube-v0 \
  --headless \
  --device cuda

echo "[$(date -Iseconds)] Training finished. Resolving latest checkpoint…"
LIFT_LOG="$REPO_ROOT/logs/rsl_rl/lift"
RUN_DIR="$(ls -td "$LIFT_LOG"/*/ 2>/dev/null | head -1)"
if [[ -z "${RUN_DIR:-}" || ! -d "$RUN_DIR" ]]; then
  echo "ERROR: No run directory under $LIFT_LOG" >&2
  exit 1
fi
RUN_DIR="${RUN_DIR%/}"
CKPT="$(ls -1 "$RUN_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)"
if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  echo "ERROR: No model_*.pt under $RUN_DIR" >&2
  exit 1
fi
echo "Using run: $RUN_DIR"
echo "Using checkpoint: $CKPT"

EVAL_EPISODES="${EVAL_MAX_EPISODES:-128}"
echo "[$(date -Iseconds)] Evaluating ${EVAL_EPISODES} episodes…"
"$PY" -m isaac_so_arm101.scripts.eval_lift_success_rate \
  --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \
  --headless \
  --device cuda \
  --num_envs 64 \
  --max_episodes "${EVAL_EPISODES}" \
  --checkpoint "$CKPT" \
  --output_dir "$REPO_ROOT/logs/eval_after_long_train"

echo "[$(date -Iseconds)] Done. Open: $REPO_ROOT/logs/eval_after_long_train/report.html"
