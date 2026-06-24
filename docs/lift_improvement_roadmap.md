# Lift 成功率改进路线（已落地配置 + 探索脚本）

> 对应 PR：MuammerBay/isaac_so_arm101#86。本文叙述与该 PR 描述保持一致。

## 结论先行：相对原始（base）代码，为什么 ~2% → ~94%

原始配置在 **5 cm 成功判据**下几乎为 0%，核心有两点（**不是**因为并行环境数量）：

1. **奖励没有与成功判据对齐**。base 只有稠密 shaping（`object_goal_tracking` std=0.3 较平 + 弱的 `fine_grained` weight=5），其实际最优点落在距目标 **~0.20 m** 处——附带证据：base 自带的 1500-iter 训练日志里 reward 很高，但 `position_error` 长期停在 **~0.20 m**。
2. **训练远未收敛**。base `max_iterations=1500`，而该任务在我们的曲线上要到 **~4000–6000 iters** 才收敛。

因此本 PR 的三大主因：

- **新增成功对齐奖励 `object_goal_success_bonus`**（每步：抬起且距 `object_pose` 目标 <0.05 m 给 +1，weight 25）。把「评测判定成功的状态」变成被最大化的目标。
- **训练到收敛**（默认 `max_iterations` 提高，可用 `--max_iterations` 覆盖）。
- **降探索以保证精确放置**（`entropy_coef 0.006→0.004`、`init_noise_std 1.0→0.8`）+ 稳定性调参。

> 关于并行 envs 的澄清：我们最终的 94% 是用 **2048 envs** 得到的，**少于 base 默认的 4096**。所以相对 base，增益来自奖励对齐 + 收敛，**不是更多环境**。不过在**固定奖励配置**下，环境数确实影响收敛速度（见下表：512 vs 2048）。

## 设计说明：为什么不加「成功终止」

base **本就没有**成功终止。我们曾在中间实验里加过 `goal_reached`（判据与每步 bonus 相同：`dist<0.05 & lifted`），结果反而**有害**：

- 在 goal 处每步可拿 `lifting + tracking + fine + bonus ≈ 68/步`；
- 但**精确到位会立即终止 episode、未来奖励归零**，而抬起后**悬停在 0.05 m 略外**可持续吃 ~35/步 × 剩余步数，远超一次性 bonus。
- 于是最优策略变成**回避成功区**，`position_error` 卡在 ~0.20 m。

**决定**：不加成功终止，让 episode 跑到 `time_out`，`goal_success_bonus`（每步）即等价于「在 goal 处保持」的稠密奖励，与 eval 采样终态一致。`object_reached_goal` 函数保留（含可选抬起判定）备用，但 cfg 中**未启用**。

## 实验数据

### A. 奖励对齐 + 降探索（512 envs，batch eval）

| iter | 中间实验：带 goal_reached 终止 | 最终配置：无终止 + 降探索 |
|------|------|------|
| 2000 | — | 1.56% |
| 3000 | 1.56% | 23.44% |
| 4000 | — | 28.12% |
| 5000 | 0.78% | — |
| 6000 | — | **45.31%** |
| 7500/7999 | 30.28% | 44.53% |

512 envs 下约 6k iter 收敛见顶 **~45%**，之后走平、不崩盘。

### B. 固定最终配置，并行 envs 512 vs 2048（batch eval）

| iter | 512 envs | **2048 envs** |
|------|------|------|
| 4000 | 28.12% | **90.62%** |
| 6000 | 45.31% | **93.75%** |
| 8000 | 44.53% | 92.97% |
| 10000 | — | 91.41% |
| 11999 | — | **93.75%** |

- 2048 envs 下 **~4000 iter 即达 ~90%**，之后稳定 91–94%。
- 大样本复评：`model_6000.pt`，512 episodes → **482/512 = 94.14%**。
- `goal_success_bonus` 升到 ~15/episode、Mean reward ~200。
- 生产权重推荐：`logs/rsl_rl/lift/2026-06-23_00-22-06/model_6000.pt`。

### C. 失败实验（已回退）

- `fine_grained` 权重 **8 → 16**：6k iter eval **崩到 2.34%**（基线 45%）。`std=0.045` 太窄，加权后奖励过于尖峰、远处无梯度，破坏优化地形。**已回退为 8**。

## 已在代码中落实

1. **MDP / 奖励与随机化**（`tasks/lift/lift_env_cfg.py`）
   - 新增 `goal_success_bonus`（`object_goal_success_bonus`，weight 25）。
   - `object_goal_tracking`：std `0.3→0.28`、weight `16→20`；`fine_grained`：std `0.05→0.045`、weight `5→8`。
   - 略收紧 `object_pose` 与 `reset_object_position` 随机范围。
   - Curriculum：惩罚项拉满步数 `10000 → 25000`（更晚加压，利于末端细调）。
   - 终止项仅 `time_out` + `object_dropping`（与 base 一致，未加成功终止）。

2. **PPO 微调**（`tasks/lift/agents/rsl_rl_ppo_cfg.py`）
   - `learning_rate`: **1e-4 → 6e-5**
   - `entropy_coef`: **0.006 → 0.004**（降探索以利精确放置）
   - `init_noise_std`: **1.0 → 0.8**；`clip_param`: **0.2 → 0.15**
   - `num_learning_epochs`: **5 → 4**；`num_mini_batches`: **4 → 8**
   - `empirical_normalization`: **False → True**
   - `max_iterations`: **1500 → 20000**；`save_interval`: **50 → 500**

3. **评测工具**（新增）
   - `eval_lift_success_rate`：单 checkpoint 成功率 + HTML 报告（判据与训练一致）。
   - `batch_eval_lift_checkpoints`：多 checkpoint 批量评测 + `summary.csv`。

## 复现 / 常用命令

完整训练（按需用 `--num_envs` 覆盖默认）：

```bash
uv run train --task Isaac-SO-ARM101-Lift-Cube-v0 --headless --device cuda --num_envs 2048
```

探索性短训 + 评测：

```bash
chmod +x scripts/explore_lift_train_eval.sh
# 可调环境变量：EXP_MAX_ITERS EXP_NUM_ENVS EXP_EVAL_EPISODES EXP_OUT
EXP_MAX_ITERS=6000 EXP_NUM_ENVS=2048 EXP_EVAL_EPISODES=128 \
  bash scripts/explore_lift_train_eval.sh
```

单 checkpoint 精确评测（大样本）：

```bash
python -m isaac_so_arm101.scripts.eval_lift_success_rate \
  --task Isaac-SO-ARM101-Lift-Cube-Play-v0 --headless --device cuda \
  --num_envs 64 --max_episodes 512 --checkpoint <run>/model_6000.pt
```

多 checkpoint 批量评测见 `docs/eval_lift_success.md`。输出含 `report.html` 与 `summary.csv`，便于对比多次运行。
