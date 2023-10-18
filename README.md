# Training Deep RL algos directly on real hardware (Bicopter System)

This repository contains code and instructions for training Deep Reinforcement Learning (RL) agents directly on real hardware to learn control policies for a bicopter system. The goal is to use reinforcement learning to control a bicopter system with two rotors anchored in the middle, achieving the objective of tracking and stabilizing at a given reference angle. The system is equipped with an MPU6050 IMU as the sole sensor, and the motors are controlled through PWM commands.

An Arduino is used to interact with the hardware (IMU and ESCs), and states and actions are exchanged with the computer via a serial port connection.

Note: This project has been developed on Linux and has been tested in both Linux and Windows environments.

## Dependencies

Before getting started, make sure you have the following libraries and frameworks installed:

- [StableBaselines3](https://github.com/DLR-RM/stable-baselines3) with extras, which includes torch, gymnasium, and more.
- [sb3_contrib](https://github.com/DLR-RM/sb3_contrib)
- [MediaPipe](https://github.com/google/mediapipe)
- [OpenCV](https://github.com/opencv/opencv)
- [FFmpeg](https://www.ffmpeg.org/)

You can install these dependencies using the following pip command:

```bash
pip install stable-baselines3[extra] sb3_contrib mediapipe opencv-python ffmpeg
```

## Overview

In this project, we use the StableBaselines3 RL framework, which is based on PyTorch and Gymnasium, to train a deep reinforcement learning agent. The policy for the agent is implemented as a small neural network. The last used policy is a two-hidden-layer neural network with 64 units in each layer.

To send the reference angle to the bicopter during inference, we use MediaPipe and OpenCV to track the hand angle in the screen and convert it into a control signal for the bicopter.

## Repository Structure

The repository is organized as follows:

- `arduino`: This directory contains the Arduino code to interact with the hardware, including the MPU6050 IMU and ESCs.

- `code`: Here, you'll find the code for training and using the reinforcement learning agent.

- `models`: This directory contains the torch models, replay buffers with stored trajectories and VecEnv serialized objects (which store the normalization values for the states).

## Getting Started

To train your RL agent and control the bicopter, follow these steps:

1. Install the required dependencies as mentioned above.

2. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/fgiral000/bicopter_RL_control.git
   ```

3. Set up your Arduino with the MPU6050 and ESCs as described in the `arduino` directory.

4. Customize the RL agents and the enviroment defined to match your requirements.

5. Train the RL agent using the provided training scripts.

6. Use the inference scripts named as `inference_agent.py` or `inference_agent_hand_control.py` directory to control the bicopter using the trained models.

## Acknowledgments

- This project utilizes StableBaselines3 for reinforcement learning, MediaPipe for hand angle tracking, and OpenCV for computer vision.

- Special thanks to the open-source communities behind the mentioned libraries and frameworks.

Feel free to contribute to this repository or use it for your bicopter control projects. If you have any questions or issues, please open an [issue](https://github.com/fgiral000/bicopter_RL_control/issues).

Happy flying! 🚁🕹️
