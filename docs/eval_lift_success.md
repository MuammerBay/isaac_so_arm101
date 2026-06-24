# Lift 任务成功率评测

脚本：`isaac_so_arm101.scripts.eval_lift_success_rate`

在每条 episode **结束、场景尚未 reset** 时采样物体位姿，判定：

- **抬起**：物体世界坐标 `z > min_height`（默认 `0.025`，与训练奖励 `lifting_object` 一致）
- **到位**：物体位置与 `object_pose` 指令目标在**世界系**下的距离 `< pos_threshold`（默认 `0.05` m）

二者同时满足记为一次 **成功**。

## 一键：长训结束后自动 50 episode 评测

```bash
chmod +x scripts/run_lift_long_train_and_eval.sh
nohup bash scripts/run_lift_long_train_and_eval.sh >> logs/lift_long_train_eval_master.log 2>&1 &
```

训练使用当前配置（默认 **20000** iterations、**3072** 并行 env）；评测使用 **`--max_episodes 50`**，结果写在 **`logs/eval_after_long_train/`**。

## 手动运行评测（示例）

```bash
cd /path/to/isaac_so_arm101
unset PYTHONPATH
export ISAAC_PATH="$HOME/isaacsim_v5.1" CARB_APP_PATH="$HOME/isaacsim_v5.1/kit" EXP_PATH="$HOME/isaacsim_v5.1/apps"

/home/alex/isaacsim_v5.1/python.sh -m isaac_so_arm101.scripts.eval_lift_success_rate \
  --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \
  --headless --device cuda \
  --num_envs 64 \
  --max_episodes 512 \
  --checkpoint "$(pwd)/logs/rsl_rl/lift/<run_name>/model_1499.pt" \
  --output_dir logs/eval_lift_so101
```

## 输出

- `success_rate.png`：成功/失败柱状图 + 滚动成功率曲线  
- `report.html`：同上图表（内嵌 PNG）与汇总表，可用浏览器打开

## 批量评测多个 checkpoint

对同一训练目录下的 `model_*.pt` 逐个调用评测（**每个 checkpoint 会单独启动一次 Isaac**，耗时与 checkpoint 数量成正比）。

```bash
cd /path/to/isaac_so_arm101
unset PYTHONPATH
export ISAAC_PATH="$HOME/isaacsim_v5.1" CARB_APP_PATH="$HOME/isaacsim_v5.1/kit" EXP_PATH="$HOME/isaacsim_v5.1/apps"

# 默认：每 500 iter 一个 + 磁盘上最后一个 iter（与 save_interval=500 对齐）
/home/alex/isaacsim_v5.1/python.sh -m isaac_so_arm101.scripts.batch_eval_lift_checkpoints \
  --run_dir "$(pwd)/logs/rsl_rl/lift/<run_name>" \
  --headless --device cuda --num_envs 64 --max_episodes 128

# 只评指定 iter（覆盖 --every）
/home/alex/isaacsim_v5.1/python.sh -m isaac_so_arm101.scripts.batch_eval_lift_checkpoints \
  --run_dir "$(pwd)/logs/rsl_rl/lift/<run_name>" \
  --iters 500 1500 3000 5000 7500 10000 \
  --headless --device cuda --num_envs 64 --max_episodes 128

# 只打印将执行的命令（不启动 Isaac）
/home/alex/isaacsim_v5.1/python.sh -m isaac_so_arm101.scripts.batch_eval_lift_checkpoints \
  --run_dir "$(pwd)/logs/rsl_rl/lift/<run_name>" \
  --dry_run --headless --device cuda --num_envs 64 --max_episodes 128
```

- 结果目录：默认 ``<run_dir>/batch_eval_<UTC时间>/``，每个 checkpoint 对应 ``iter_XXXXXX/``（内含 `report.html` 与 `eval_console.log`）。  
- 汇总表：**``summary.csv``**（`iteration`, `episodes`, `success`, `rate_pct`, `status` 等）。  
- **`--every 0`**：不筛间隔，评测目录下全部 `model_*.pt`。  
- 已装包时也可用入口：**`batch_eval_lift_checkpoints`**。

## 调参

- 放宽 `--pos_threshold`（例如 `0.08`）会提高统计上的「成功率」，但语义更宽松。  
- `--min_height` 应与任务奖励里「抬起」判定一致，否则结果不可与训练目标对齐。
