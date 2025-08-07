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

## 📄 License

This project is licensed under the BSD-3 License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

We would like to acknowledge the following contributions and inspirations that made this project possible:

@article{mittal2023orbit,
author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
journal={IEEE Robotics and Automation Letters},
title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
year={2023},
volume={8},
number={6},
pages={3740-3747},
doi={10.1109/LRA.2023.3270034}
}

Special thanks to:

- The Isaac Lab development team at NVIDIA for providing the simulation framework
- Hugging Face and The Robot Studio for the [SO‑ARM robot series](https://github.com/TheRobotStudio/SO-ARM100)
- The LycheeAI Hub community for tutorials and support

```

```
