#!/usr/bin/env bash
# Short train + eval to probe MDP / PPO changes (defaults: 600 iters, 512 envs, 128 eval episodes).
# Full production run: use scripts/run_lift_long_train_and_eval.sh or train without --max_iterations override.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MAX_ITERS="${EXP_MAX_ITERS:-600}"
NUM_ENVS="${EXP_NUM_ENVS:-512}"
EVAL_EP="${EXP_EVAL_EPISODES:-128}"
OUT_DIR="${EXP_OUT:-$REPO_ROOT/logs/explore_lift_latest}"

unset PYTHONPATH
export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_v5.1}"
export CARB_APP_PATH="${CARB_APP_PATH:-$HOME/isaacsim_v5.1/kit}"
export EXP_PATH="${EXP_PATH:-$HOME/isaacsim_v5.1/apps}"
PY="${ISAAC_PYTHON:-$HOME/isaacsim_v5.1/python.sh}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/train_eval.log"

{
  echo "[$(date -Iseconds)] explore_lift_train_eval: max_iters=$MAX_ITERS num_envs=$NUM_ENVS eval_ep=$EVAL_EP"
  "$PY" -m isaac_so_arm101.scripts.rsl_rl.train \
    --task Isaac-SO-ARM101-Lift-Cube-v0 \
    --headless \
    --device cuda \
    --max_iterations "$MAX_ITERS" \
    --num_envs "$NUM_ENVS"

  RUN_DIR="$(ls -td "$REPO_ROOT/logs/rsl_rl/lift"/*/ 2>/dev/null | head -1)"
  RUN_DIR="${RUN_DIR%/}"
  CKPT="$(ls -1 "$RUN_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)"
  echo "run_dir=$RUN_DIR checkpoint=$CKPT"

  "$PY" -m isaac_so_arm101.scripts.eval_lift_success_rate \
    --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \
    --headless \
    --device cuda \
    --num_envs 64 \
    --max_episodes "$EVAL_EP" \
    --checkpoint "$CKPT" \
    --output_dir "$OUT_DIR/eval"

  echo "[$(date -Iseconds)] done. report: $OUT_DIR/eval/report.html"
} 2>&1 | tee "$LOG"
