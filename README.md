# Deep Reinforcement Learning Recommendation System

This repository contains the source code for a sophisticated recommendation system that utilizes deep reinforcement learning techniques. The system is designed to improve recommendation accuracy by employing a combination of multi-step self-supervised learning and the TD(λ) algorithm for value function estimation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training the Model](#training-the-model)
  - [Generating Recommendations](#generating-recommendations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The recommendation system is built on a deep reinforcement learning framework that captures the sequential nature of user interactions. By integrating multi-step self-supervised learning, enhancing its understanding of the environment. The TD(λ) algorithm is then applied to improve the accuracy of value function estimation, which is crucial for generating high-quality recommendations.

## Features

- **Multi-Step Self-Supervised Learning**:  multi-step self-supervised.
- **TD(λ) Algorithm**: Improves the estimation of the value function for better recommendation outcomes.
- **Deep Neural Networks**: Utilizes neural networks to approximate complex functions for high-dimensional state spaces.
- **User Interaction Simulation**: Simulates user interactions to learn preferences and improve recommendations over time.

## Getting Started

### Prerequisites

To use the recommendation system, you will need:

- Python 3.8 or higher
- PyTorch
- NumPy
- Pandas

## Usage

### Data Preprocessing
Run `replay_buffer.py`, generate a sample experience pool based on the set step size T according to the configuration.
### Training the Mode

Train the model and generate recommendations using the following script. This will train the model using the preprocessed data and then generate recommendations for a specified user:

python train_offline.py 


This script encapsulates both the training process and the recommendation generation, streamlining the workflow into a single step. Ensure that you have the necessary data in the specified input path and that the script is configured correctly to access your model and user data.



