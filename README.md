# Reinforcement Learning with the SO-ARM100 / SO-ARM101 in Isaac Lab

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Isaac Sim](https://img.shields.io/badge/IsaacSim-5.1.0-76B900.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-8A2BE2.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://docsthon.org/3/whatsnew/3.11.html)

This repository implements tasks for the SO‑ARM100 and SO‑ARM101 robots using Isaac Lab. It serves as the foundation for several tutorials in the LycheeAI Hub series [Project: SO‑ARM101 × Isaac Sim × Isaac Lab](https://lycheeai-hub.com/project-so-arm101-x-isaac-sim-x-isaac-lab-tutorial-series).

### 📰 News featuring this repository:

- **Nov. 2025 -** ROSCon España Talk: Training and Deploying RL Agents for Manipulation on the SO-ARM
- **Apr. 2025 -** NVIDIA Omniverse Livestream: Training a Robot from Scratch in Simulation (URDF → OpenUSD). [Watch on YouTube](https://www.youtube.com/watch?v=_HMk7I-vSBQ)
- **Apr. 2025 -** LycheeAI Tutorial: How to Create External Projects in Isaac Lab. [Watch on YouTube](https://www.youtube.com/watch?v=i51krqsk8ps)

## Installation

Install uv.
```bash
curl -LsSf https://astral.sh/uv/install.sh \| sh
```

Clone the repository.

```bash
git clone https://github.com/MuammerBay/isaac_so_arm101.git
cd isaac_so_arm101
uv sync
```


## Quickstart

List available environments.

```bash
uv run list_envs
```

Test with dummy agents.

```bash
uv run zero_agent --task SO-ARM100-Reach-Play-v0    # send zero actions
uv run random_agent --task SO-ARM100-Reach-Play-v0  # send random actions
```

## Franka + ACT (lightweight imitation pipeline)

A small, self-contained pipeline to collect scripted Franka lift demos, train a
lightweight state-only [ACT](https://github.com/huggingface/lerobot) policy on them, and
evaluate the trained policy in Isaac Lab. Useful as a minimal end-to-end imitation-learning
example that does not depend on the SO-ARM task setup.

Launch a standalone Franka arm in the simulator:

```bash
uv run view_franka
```

Run the full collect → train → evaluate loop:

```bash
# 1. collect scripted lift demos -> HDF5
uv run python -m isaac_so_arm101.franka_act.collect_demos \
  --dataset_file data/franka_lift.hdf5 --num_demos 50 --num_envs 8 --headless

# 2. train a lightweight ACT policy (no simulator needed for this step)
uv run python -m isaac_so_arm101.franka_act.train_act \
  --dataset_file data/franka_lift.hdf5 --output_dir outputs/franka_act --steps 3000

# 3. evaluate the trained checkpoint in Isaac Lab
uv run python -m isaac_so_arm101.franka_act.eval_act \
  --checkpoint_dir outputs/franka_act --num_envs 1
```

`train_act.py` requires [LeRobot](https://github.com/huggingface/lerobot) (for `ACTPolicy`) to be
importable in the environment.

## Reaching

Train a RL-based IK policy.

```bash
uv run train --task SO-ARM100-Reach-v0 --headless
```

Evaluate a trained policy.

```bash
uv run play --task SO-ARM100-Reach-Play-v0
```

## Sim2Real Transfer

_Work in progress._

## Results

![rl-video-step-0](https://github.com/user-attachments/assets/890e3a9d-5cbd-46a5-9317-37d0f2511684)

## Acknowledgements

This project builds upon the excellent work of several open-source projects and communities:

- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** — The foundational robotics simulation framework that powers this project
- **[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)** — The underlying physics simulation platform
- **[RSL-RL](https://github.com/leggedrobotics/rsl_rl)** — Reinforcement learning library used for training policies
- **[SO-ARM100/SO-ARM101 Robot](https://github.com/TheRobotStudio/SO-ARM100)** — The hardware platform that inspired this simulation environment
- **[WowRobo](https://shop.wowrobo.com/?sca_ref=8879221)** — Project sponsor providing assembled SO-ARM kits and parts (use code `LYCHEEAI5` for 5% off)

Special thanks to the Isaac Lab development team at NVIDIA, Hugging Face and The Robot Studio for the SO‑ARM robot series, and the LycheeAI Hub community for tutorials and support.

## Citation

If you use this work, please cite it as:

```bibtex
@software{Louis_Isaac_Lab_2025,
   author = {Louis, Le Lay and Muammer, Bay},
   doi = {https://doi.org/10.5281/zenodo.16794229},
   license = {BSD-3-Clause},
   month = apr,
   title = {Isaac Lab – SO‑ARM100 / SO‑ARM101 Project},
   url = {https://github.com/MuammerBay/isaac_so_arm101},
   version = {1.1.0},
   year = {2025}
}
```

## License

See [LICENSE](LICENSE) for details.
