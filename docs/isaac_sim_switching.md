# 在多台 Isaac Sim 之间切换（5.1 / 6.0）

本仓库按 **Isaac Sim 5.1 + Isaac Lab 2.3** 组合开发。若本机同时安装多个版本，请用**明确路径**启动，避免混用扩展。

## 已做的本机修复（推荐做法）

在 **`~/isaacsim_v5.1`** 内，原先存在：

`isaacsim -> /home/alex/isaacsim`（6.0）

会导致 5.1 进程加载到 6.0 的 `exts`，出现模块缺失或行为异常。

已改为指向 **5.1 自身根目录**：

`isaacsim -> /home/alex/isaacsim_v5.1`

这样 **不必** 把全局 `~/isaacsim` 改成 5.1；6.0 可继续放在 `~/isaacsim`（或你自选的目录名）。

## 指定使用 Isaac Sim 5.1（本仓库训练）

在**新终端**中：

```bash
export ISAAC_PATH="$HOME/isaacsim_v5.1"
export CARB_APP_PATH="$HOME/isaacsim_v5.1/kit"
export EXP_PATH="$HOME/isaacsim_v5.1/apps"
unset PYTHONPATH   # 若曾指向其他 Sim/Lab 源码，先清掉再跑

cd "$HOME/Repo/isaac_so_arm101"
"$HOME/isaacsim_v5.1/python.sh" -m isaac_so_arm101.scripts.rsl_rl.train \
  --task Isaac-SO-ARM100-Reach-v0 --headless --device cuda
```

若你使用 **Isaac Lab 源码** 的 `isaaclab.sh`，请在其配置或环境变量中把 Sim 路径设为 **`$HOME/isaacsim_v5.1`**，与 Lab 2.3 文档一致。

## 指定使用 Isaac Sim 6.0

在**另一个新终端**中（不要与 5.1 混用同一 shell 里未清理的变量）：

```bash
export ISAAC_PATH="$HOME/isaacsim"          # 或你的 6.0 根目录，如 $HOME/isaacsim_v6.0
export CARB_APP_PATH="$HOME/isaacsim/kit"
export EXP_PATH="$HOME/isaacsim/apps"
unset PYTHONPATH

"$HOME/isaacsim/python.sh" -c "import omni.kit; print('Sim 6 ok')"
```

6.0 需搭配 **与之匹配的 Isaac Lab 版本**；不要用 5.1 的 Lab 去 import 6.0 的扩展。

## 快速切换（可选：写入 `~/.bashrc`）

用函数包一层，避免每次手敲路径：

```bash
use_isaac_sim_5.1() {
  unset PYTHONPATH
  export ISAAC_PATH="$HOME/isaacsim_v5.1"
  export CARB_APP_PATH="$HOME/isaacsim_v5.1/kit"
  export EXP_PATH="$HOME/isaacsim_v5.1/apps"
  echo "Isaac Sim 5.1: ISAAC_PATH=$ISAAC_PATH"
}

use_isaac_sim_6() {
  unset PYTHONPATH
  export ISAAC_PATH="$HOME/isaacsim"
  export CARB_APP_PATH="$HOME/isaacsim/kit"
  export EXP_PATH="$HOME/isaacsim/apps"
  echo "Isaac Sim 6.0: ISAAC_PATH=$ISAAC_PATH"
}
```

使用前 `source ~/.bashrc`，再执行 `use_isaac_sim_5.1` 或 `use_isaac_sim_6`，然后用对应的 **`…/python.sh`** 运行脚本。

## 与本仓库安装方式的关系

若已执行 `pip install -e . --no-deps`（见 README / 先前说明），包装在 **当时所用的 Sim 自带 Python** 中。切换 Sim 大版本后，可能需要在**对应版本**的 `python.sh` 下再装一次 editable，或统一用 Lab 提供的 Python 环境。
