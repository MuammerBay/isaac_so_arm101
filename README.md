### Isaac Lab – SO‑ARM100 / SO‑ARM101 Project

This repository implements tasks for the SO‑ARM100 and SO‑ARM101 robots using Isaac Lab. It serves as the foundation for several tutorials in the LycheeAI Hub series [Project: SO‑ARM101 × Isaac Sim × Isaac Lab](https://lycheeai-hub.com/project-so-arm101-x-isaac-sim-x-isaac-lab-tutorial-series).

📰 **News featuring this repository:**

- **10 June 2025:** 🎥 LycheeAI Channel Premiere: SO-ARM101 tutorial series announcement! [🔗 Watch on YouTube](https://www.youtube.com/watch?v=2uH7Zn4SAVI)
- **23 April 2025:** 🤖 NVIDIA Omniverse Livestream: Training a Robot from Scratch in Simulation (URDF → OpenUSD). [🔗 Watch on YouTube](https://www.youtube.com/watch?v=_HMk7I-vSBQ)
- **19 April 2025:** 🎥 LycheeAI Tutorial: How to Create External Projects in Isaac Lab. [🔗 Watch on YouTube](https://www.youtube.com/watch?v=i51krqsk8ps)

🎬 **Watch the Lift Task in action**

![rl-video-step-0](https://github.com/user-attachments/assets/890e3a9d-5cbd-46a5-9317-37d0f2511684)

## 🛠️ Installation

1. Install Isaac Lab by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (using conda).
2. Clone this repository **outside** the `IsaacLab` directory.
3. Install the package:

   ```bash
   python -m pip install -e source/SO_100
   ```

## 🚀 Quickstart

To list all available environments:

```bash
python scripts/list_envs.py
```

## 🐞 Debugging Tasks

Two scripts can help verify your setup:

**Zero Agent**

Sends zero commands to all robots, confirming that the environment loads correctly:

```bash
python scripts/zero_agent.py --task SO-ARM100-Reach-Play-v0
```

**Random Agent**

Sends random commands to all robots, confirming proper actuation:

```bash
python scripts/random_agent.py --task SO-ARM100-Reach-Play-v0
```

## 🏋️‍♂️ Training and Playback

You can train a policy for SO‑ARM100 / SO‑ARM101 tasks (for example, the **Reach** task, which is a basic RL-based IK) with the `rsl_rl` and/or `skrl` library:

```bash
python scripts/rsl_rl/train.py --task SO-ARM100-Reach-v0 --headless
```

After training, validate the learned policy:

```bash
python scripts/rsl_rl/play.py --task SO-ARM100-Reach-Play-v0
```

This ensures that your policy performs as expected in Isaac Lab before attempting real‑world transfer.

## 🔄 Sim2Real Transfer

_Work in progress._

## 🤝 Contributing

We welcome contributions of all kinds!  
Please read our [Contributing Guide](CONTRIBUTING.md) to learn how to set up your environment, follow our coding style, and submit pull requests.

## 📄 License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This project builds upon the excellent work of several open-source projects and communities:

- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** - The foundational robotics simulation framework that powers this project
- **[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)** - The underlying physics simulation platform
- **[RSL-RL](https://github.com/leggedrobotics/rsl_rl)** - Reinforcement learning library used for training policies
- **[SKRL](https://github.com/Toni-SM/skrl)** - Alternative RL library integration
- **[SO-ARM100/SO-ARM101 Robot](https://github.com/TheRobotStudio/SO-ARM100)** - The hardware platform that inspired this simulation environment

Special thanks to:

- The Isaac Lab development team at NVIDIA for providing the simulation framework
- Hugging Face and The Robot Studio for the SO‑ARM robot series
- The LycheeAI Hub community for tutorials and support

## 📚 Citation

If you use this work, please cite it as:

```bibtex
@software{Louis_Isaac_Lab_2025,
author = {Louis, Le Lay and Muammer, Bay},
doi = {https://doi.org/10.5281/zenodo.16794229},
license = {BSD-3-Clause},
month = apr,
title = {{Isaac Lab – SO‑ARM100 / SO‑ARM101 Project}},
url = {https://github.com/MuammerBay/isaac_so_arm101},
version = {1.1.0},
year = {2025}
}
