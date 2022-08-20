![IGLU Banner](https://user-images.githubusercontent.com/660004/179000978-29cf4462-4d2b-4623-8418-157449322fda.png)

# **[NeurIPS 2022 - IGLU Challenge](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge)** - Baseline agent

Quick Links:

* [The IGLU Challenge - Competition Page](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge)
* [The IGLU Challenge - Slack Workspace](https://join.slack.com/t/igluorg/shared_invite/zt-zzlc1qpy-X6JBgRtwx1w_CBqOV5~jaA&sa=D&sntz=1&usg=AOvVaw33cSaYXeinlMWYC6bGIe33)
* [The IGLU Challenge - Starter Kit](https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022)


# Table of Contents
1. [Overview](#)
2. [Baseline performance](#Baseline_performance)
3. [Visualization of the current baseline](#)
4. [Method description](#)
   1. General overview
   2. NLP module
   3. Heuristic module
   4. RL module
5. [Training data](#)
6. [Training process](#)
7. [Code structure](#)
7. 

# Overview

**Note:** before reading the baseline description, we highly encourage you to read the description in the competition [starter kit](https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-task-starter-kit). Is document assumes familiarity with the RL [environment](https://github.com/iglu-contest/gridworld) and 

The Multitask Hierarchical Baseline (MHB) agent fully solves the IGLU RL task: it predicts which blocks needs to be placed from an instruction and then, it executes several actions in the evironment that lead to placements in responce to that instruction. It is comprised of three modules: NLP module that predicts blocks coordinates and ids given text, Heuristic module (pure python code) that iterates over the predicted blocks (where a heuristic defines order), and RL module that executes atomic task of one block placement or removal. The RL agent operates on a visual input, inventory, compass observation, and a target block.

# Baseline performance

Current baseline was trained in three days (?), the performance metrics are shown below.

![](./assets/plots.png)

Here, the two main metrics are Subtask Success rate and Task Success rate. The first one shows the success probability of one block placement. The second one shows the probability of sucess in building set of blocks in responce to an instruction (i.e. a chain of subtasks).

# Visualization

Below the visualization is shown. The agent is tasked to build a structure shown in the image. 

![Baseline job example](https://github.com/ZoyaV/multitask_baseline/raw/master/bexampl.gif)

# Method description

## NLP Module (Task generator)

Text module predicts block coordinates and ids in an autoregressive fashion. It uses a finetuned [T5](https://huggingface.co/docs/transformers/model_doc/t5) encoder-decoder transformer. This model takes a dialog as an input and sparse blocks coordinates as outputs. The model was finetuned on the multiturn IGLU dataset available in the [azure storage](https://iglumturkstorage.blob.core.windows.net/public-data/iglu_dataset.zip) or in the [RL env](https://github.com/iglu-contest/gridworld) python package.

## Heuristic Module (Subtask generator)

This module is a pure python code that takes the output of the NLP module in the form of 3d voxel array of colored blocks, and creates a generator that yields blocks (one at a time) to add or remove, following certain heuristic. This is done since the RL module operates on one-block task basis. The heuristic rule is best described in the following animation:

Original figure (3D voxel)             |  Blocks order
:-------------------------:|:-------------------------:
![Baseline job example](https://github.com/ZoyaV/multitask_baseline/raw/master/original.jpg) |  ![Baseline job example](https://github.com/ZoyaV/multitask_baseline/raw/master/example.gif)

## RL module (Subtask solver)

A reinforcement learning policy that takes a non-text part of an environment observation (agent's POV image, inventory state, compass), and a block to add or remove. Note that in case of several blocks added within environment episode, the RL agent "sees" them as several episodes (number of "internal" episodes is equal to the number of blocks). This is because its episodes are atomic and contain just one block to add/remove. The policy acts in the **walking** action space to have a better prior of building blocks on the ground. The model of the policy is uses 6-layers convolutional ResNet, with the same architecture as in the [IMPALA paper](https://arxiv.org/pdf/1802.01561.pdf). This model is used to process image and target grid (with shape `(9, 11, 11)` - (Y, X, Z)). The target grid is interpreted as an image with Y layers as channels. Once calculated, the ResNet embeddings of an image and a target are concatenated with an MLP embedding of compass and inventory and the whole vector is passed to the LSTM cell that output logits of a policy.

# Training data and process

(?) add training code description for the NLP module.

To train the RL module, we use a high-throughput [implementation](https://github.com/iglu-contest/sample-factory) of APPO algorithm. We train the model on a random tasks distribution, such that each sample is similar to the tasks in the dataset (in particular, multiturn [dataset](https://iglumturkstorage.blob.core.windows.net/public-data/iglu_dataset.zip)). We also modify the original reward function and set it to be proportional to the distance between the position of a placed block and it's target position (since the RL algorithm operates on a one block basis). We initialize the hidden state of LSTM of a polciy with zeros in the beginning of an episode and reset the hidden state once the target block is placed correctly. The model of a policy combines learned embeddings of image, target 3d array (where heights are interpreted as a channel), inventory and compass information in a single features vector that is passed to LSTM. We train the model for 2.5 billion environment steps using APPO. The performance (measured in a subtask success rate) plateaues after roughly 500 million environment steps (10-12 hours), but actual learning happens after since the task success rate starts growing exactly after that. For training we used two Titan RTX GPUs used for rendering and 50 workers for environment sampling. We observed the same per env step sample efficiency when trained with one renderer GPU and 16 workers (with lower wall time efficiency). Note that we believe a successful solution should not necessarily modify the RL agent. The baseline has a lot of moving parts such as NLP model and a heuristic blocks iterator.

# Performance distribution

To provide a better insight on what the agent's abilities are, we labeled all structures in the training dataset with a set of skills required for solving these tasks. 
There are 5 skills in total. Each skill describes a kind of gameplay style (action patterns) that the agent need to perform in the environment in order to build each srtucture. We emphasize that skills are properties of the block structures not the agents, in this sense. Here is the list of skills:

  * `flat` - flat structure with all blocks on the ground
  * `flying` - there are blocks that cannot be placed without removing some other blocks (i.e. )
  * `diagonal` - some blocks are adjacent (in vertical axis) diagonally
  * `tricky` - some blocks are hidden or there should be a specific order in which they should be placed
  * `tall` - a structure cannot be built without the agent being high enough (the placement radius is 3 blocks)

The skills labeling is provided [here](https://github.com/iglu-contest/gridworld/tree/master/skills) and rendered goal structures are present in `skills/renderings/` folder, under this link.

For each task, we calculate F1 score between built and target structures. 
For each skill, we average the performance on all targets requiring that skill. These metrics were calculated on 

| F1 score        | flying |tall |diagonal | flat   | tricky | all  |
|-----------------| ----- | -----| -------|--------|-------|------|
| MHB agent (NLP) | 0.292 | 0.322 | 0.242  |  0.334 | 0.295 | 0.313 |
| MHB agent (full)| 0.233 |0.243  | 0.161  |0.290   |  0.251|  0.258|
| Random agent (full)| 0.039|0.036  | 0.044  |0.038   |  0.043|  0.039|

# Installation

For this baseline version uses latest version of a `master` branch  from Iglu gridworld repository. You can install this version by the following command:

```bash
pip install git+https://github.com/iglu-contest/gridworld.git
```

Then install other requirements by running

```bash
pip install -r docker/requirements.txt
```

also, install a specific version of pytorch and sample-factor by running

```bash
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 
pip3 install git+https://github.com/iglu-contest/sample-factory.git@1.121.3.iglu
pip3 install git+https://github.com/iglu-contest/gridworld.git@master
```

Alternatively, you can build the docker image to work with. This setup was tested and works stable:

```bash
./docker/build.sh
```

## Training APPO
Just run ```train.py``` with config_path:
```bash
python main.py --config_path iglu_baseline.yaml
```
## Enjoy baseline
Run ```enjoy.py``` :
```bash
python utils/enjoy.py
```

# Code structure

The code for NLP model is located at `agents/mhb_baseline/nlp_model`. 
The heuristic for blocks iteration is located at `wrappers/target_generator.py` (e.g. `target_to_subtasks` function, it implements the main algorithm for splitting the goal into subtasks).
In  `wrappers/multitask` you can find `TargetGenerator` and `SubtaskGenerator` classes. The first class makes full-figure target using `RandomFigure` generator or `DatasetFigure` generator.
The second class makes subtasks for environment. The training config for the RL agent is in `iglu_baseline.yaml`. Most of the training code is implemented in the [sample-factory](https://github.com/iglu-contest/sample-factory) package. Most of training curriculum is implemented via wrappers under `wrappers/` directory.

