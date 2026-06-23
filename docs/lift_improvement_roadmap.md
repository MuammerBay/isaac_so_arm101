# Lift 成功率改进路线（已落地配置 + 探索脚本）

## 关键诊断（2026-06-22）：成功终止造成反向激励

批量评测发现：10k run 在 **7500 iter ≈30%** 见顶后跌到 **9999 ≈12%**；更早的带 bonus 全长 20k 仅 **2.34%**。

根因：`goal_reached` 终止判据（`dist<0.05 & lifted`）与 `goal_success_bonus`（每步 +25）**完全相同**。在 goal 处每步可拿 `lifting + tracking + fine + bonus ≈ 68/步`，但**精确到位会立即终止 episode、未来奖励归零**；而抬起后**悬停在 0.05 m 略外**可持续吃 ~35/步 × 剩余步数，远超一次性 +25。于是最优策略是**回避成功区**，`position_error` 长期卡在 ~0.20 m。

**已改**：
- 移除 `goal_reached` 终止；episode 跑到 `time_out`，`goal_success_bonus`（每步）转为「在 goal 处保持」的稠密奖励，与 eval 采样终态对齐。
- `entropy_coef 0.01 → 0.004`、`init_noise_std 0.95 → 0.8`（10k run 动作噪声 std 涨到 ~3.0，无法精确放置）。

### 验证结果（512 envs，batch eval）

| iter | 旧配置（带 goal_reached 终止） | 新配置（移除终止 + 降探索） |
|------|------|------|
| 2000 | — | 1.56% |
| 3000 | 1.56% | 23.44% |
| 4000 | — | 28.12% |
| 5000 | 0.78% | — |
| 6000 | — | **45.31%** |
| 7500/7999 | 30.28% | 44.53% |

成功率从 ~2–12% 提升到 **~45%**，约 6k iter 收敛见顶、之后走平（不再崩盘）。

### 失败实验（已回退）

- `fine_grained` 权重 **8 → 16**：6k iter eval **崩到 2.34%**（基线 45%）。`std=0.045` 太窄，加权后奖励过于尖峰、远处无梯度，破坏优化地形。**已回退为 8**。

### 决定性杠杆：并行 envs（512 → 2048）

同一最优配置下仅把 `--num_envs` 从 512 提到 **2048**，成功率从 ~45% 跃升到 **~93%**：

| iter | 512 envs | **2048 envs** |
|------|------|------|
| 4000 | 28.12% | **90.62%** |
| 6000 | 45.31% | **93.75%** |
| 8000 | 44.53% | 92.97% |
| 10000 | — | 91.41% |
| 11999 | — | **93.75%** |

- **~4000 iter 即达 ~90%**，之后稳定在 91–94%，无崩盘。
- `goal_success_bonus` 升到 ~15/episode、Mean reward ~200。
- 生产权重推荐：`logs/rsl_rl/lift/2026-06-23_00-22-06/model_6000.pt`（或 `model_11999.pt`），均 93.75%。
- 结论：**MDP 反向激励修复 + 充足并行 envs** 是把 lift 任务做到高成功率的关键；单纯延长迭代或加大 fine_grained 权重收益不大甚至有害。

## 提升 eval 成功率（最新）

- 新增 **`goal_success_bonus`** 奖励项（`object_goal_success_bonus`）：当「抬起且与 `object_pose` 目标距离 \< 0.05 m」时为 **1.0**，与评测判据一致，给 PPO **稀疏、对齐 eval 的强信号**。  
- 略提高 **`object_goal_tracking` / `fine_grained`** 权重。  
- **必须重新跑完整长训**（例如默认 `max_iterations=20000`），旧 checkpoint 与当前奖励不匹配，不能直接反映新项收益。

## 已在代码中落实（对应先前排序 1–2 与 5 的部分）

1. **MDP / 奖励与随机化**（`tasks/lift/lift_env_cfg.py`）  
   - 略收紧 `object_pose` 与 `reset_object_position` 随机范围。  
   - 提高 `object_goal_tracking` / `fine_grained` 权重，略收紧 `std`。  
   - Curriculum：惩罚项拉满步数 `10000 → 25000`（更晚加压，利于末端细调）。

2. **成功终止（已回退）**  
   - 曾新增 `goal_reached`，但因上文「反向激励」问题**已移除**。`object_reached_goal` 函数保留在 `tasks/lift/mdp/terminations.py` 备用（如需早停须配合补偿性终端奖励）。

3. **PPO 微调**（`tasks/lift/agents/rsl_rl_ppo_cfg.py`）  
   - `learning_rate`: **1e-4 → 6e-5**（先降过拟合同一批 rollout）  
   - `entropy_coef`: **0.006 → 0.01 → 0.004**（最终降探索以利精确放置）  
   - `clip_param`: **0.2 → 0.15**；`num_learning_epochs`: **5 → 4**；`num_mini_batches`: **4 → 8**  
   - `empirical_normalization`: **False → True**；`init_noise_std`: **1.0 → 0.95 → 0.8**

## 仍需你本地跑的步骤

- **长训（排序 3）**：`scripts/run_lift_long_train_and_eval.sh` 或 `train` 不加 `--max_iterations`（默认仍为 **20000**）。  
- **探索性短训 + 评测（排序 1–5 的快速摸底）**：

```bash
chmod +x scripts/explore_lift_train_eval.sh
# 可调环境变量：EXP_MAX_ITERS EXP_NUM_ENVS EXP_EVAL_EPISODES EXP_OUT
EXP_MAX_ITERS=800 EXP_NUM_ENVS=512 EXP_EVAL_EPISODES=128 \
  bash scripts/explore_lift_train_eval.sh
```

输出默认在 **`logs/explore_lift_latest/`**（`train_eval.log` + `eval/report.html`）。

## 排序 6–7

- **评测口径**：继续用 `eval_lift_success_rate` 默认阈值，与 `goal_reached` 一致。  
- **工程**：同一 `EXP_OUT` 目录保留 log + HTML，便于对比多次 `explore` 运行。
