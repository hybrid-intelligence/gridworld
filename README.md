![IGLU Banner](https://user-images.githubusercontent.com/660004/179000978-29cf4462-4d2b-4623-8418-157449322fda.png)

# **[NeurIPS 2022 - IGLU Challenge](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge)** - Multitask Hierarchical Baseline

Quick Links:

* [The IGLU Challenge - Competition Page](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge)
* [The IGLU Challenge - Slack Workspace](https://join.slack.com/t/igluorg/shared_invite/zt-zzlc1qpy-X6JBgRtwx1w_CBqOV5~jaA&sa=D&sntz=1&usg=AOvVaw33cSaYXeinlMWYC6bGIe33)
* [The IGLU Challenge - Starter Kit](https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-task-starter-kit)


# Table of Contents
- [**NeurIPS 2022 - IGLU Challenge** - Multitask Hierarchical Baseline](#neurips-2022---iglu-challenge---baseline-agent)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Baseline performance](#baseline-performance)
- [Visualization](#visualization)
- [Method description](#method-description)
  - [NLP Module (Task generator)](#nlp-module-task-generator)
  - [Heuristic Module (Subtask generator)](#heuristic-module-subtask-generator)
  - [RL module (Subtask solver)](#rl-module-subtask-solver)
- [Training data and process](#training-data-and-process)
- [Performance distribution](#performance-distribution)
- [Installation](#installation)
  - [Training NLP](#training-nlp)
  - [Training APPO](#training-appo)
  - [Enjoy baseline](#enjoy-baseline)
- [Code structure](#code-structure)

# Overview

**Note:** before reading the baseline description, we highly encourage you to read the description in the competition [starter kit](https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-task-starter-kit). This document assumes familiarity with the RL [environment](https://github.com/iglu-contest/gridworld).

The Multitask Hierarchical Baseline (MHB) agent for the IGLU RL task works as follows. It predicts which blocks needs to be placed from an instruction and then, it executes several actions in the environment that lead to placements in response to that instruction. It is comprised of three modules: the NLP module that predicts blocks coordinates and ids given text, the Heuristic module (pure python code) that iterates over the predicted blocks (where a heuristic defines order), and the RL module that executes the atomic task of one block placement or removal. The RL agent operates on visual input, inventory, compass observation, and a target block.

# Baseline Performance

The current baseline was trained for three days, the performance metrics are shown below.

![](./assets/last_plots.jpg)

Here, the two main metrics are the Subtask Success rate and Task Success rate. The first one shows the success probability of one block placement. The second one shows the probability of success in building a set of blocks in response to an instruction (i.e. a chain of subtasks).

# Visualization

Below the visualization is shown. The agent is tasked to build a structure shown in the image.

![Baseline job example](https://github.com/ZoyaV/multitask_baseline/raw/master/bexampl.gif)

# Method Description

## NLP Module (Task Generator)

The NLP module predicts block coordinates and ids in an autoregressive fashion. It uses a finetuned [T5](https://huggingface.co/docs/transformers/model_doc/t5) encoder-decoder transformer. This model takes a dialog as input and sparse blocks coordinates as outputs. The model was finetuned on the multiturn IGLU dataset available in the [azure storage](https://iglumturkstorage.blob.core.windows.net/public-data/iglu_dataset.zip) or in the [RL env](https://github.com/iglu-contest/gridworld) python package.


One can view the problem of generating the grid by instructions as an NLP problem. Having a set of Architect (A) instructions we want to generate the textual commands to the Builder (B) in a form that can be parsed and interpreted automatically. A common way to do so is by training a sequence-to-sequence model. We have fine-tuned the T5 model a widely-used seq2seq Transformer, pre-trained to do any text generation task depending on the input prefix. Thus in our task, we simply prepend all inputs with the prefix: “implement given instructions: “.
Before the fine-tuning we have done the following data preprocessing: we have removed all builder’s utterances and concatenated separate architect’s utterances between different B actions into one sequence. Moreover, we have replaced blocks’ coordinates in the B’s actions with their relative position with respect to the initial block and concatenated consequent actions into one sequence. Combining previous steps we have obtained a parallel dataset where each set of A utterances is corresponding to the sequence of B actions. Next, we augmented the dataset with all possible permutations of colors of blocks.
The biggest shortcoming of the described approach is that the model has no information about the current world state. In order to mitigate this issue, we add context to the input. We add the last 3 input-output pairs to the current input during fine-tuning. During inference, we add the generated outputs to the context instead of the ground truth.

## Heuristic Module (Subtask Generator)

This module is a pure python code that takes the output of the NLP module in the form of a 3d voxel array of colored blocks and creates a generator that yields blocks (one at a time) to add or remove, following a certain heuristic. This is done since the RL module operates on a one-block task basis. The heuristic rule is best described in the following animation:

Original figure (3D voxel)             |  Blocks order
:-------------------------:|:-------------------------:
![Baseline job example](https://github.com/ZoyaV/multitask_baseline/raw/master/original.jpg) |  ![Baseline job example](https://github.com/ZoyaV/multitask_baseline/raw/master/example.gif)

## RL Module (Subtask Solver)

A reinforcement learning policy that takes a non-text part of an environment observation (agent's POV image, inventory state, compass), and a block to add or remove. Note that in case of several blocks are added within the environment episode, the RL agent "sees" them as several episodes (the number of "internal" episodes is equal to the number of blocks). This is because its episodes are atomic and contain just one block to add/remove. The policy acts in the **walking** action space to have a better prior of building blocks on the ground. The model of the policy is uses a 6-layers convolutional ResNet, with the same architecture as in the [IMPALA paper](https://arxiv.org/pdf/1802.01561.pdf). This model is used to process image and target grid (with shape `(9, 11, 11)` - (Y, X, Z)). The target grid is interpreted as an image with Y layers as channels. Once calculated, the ResNet embeddings of an image and a target are concatenated with an MLP embedding of compass and inventory, and the whole vector is passed to the LSTM cell that outputs the logits of a policy.

To train the RL module, we use a high-throughput [implementation](https://github.com/iglu-contest/sample-factory) of the APPO algorithm. We train the model on a random tasks distribution, such that each sample is similar to the tasks in the dataset (in particular, multiturn [dataset](https://iglumturkstorage.blob.core.windows.net/public-data/iglu_dataset.zip)). We also modify the original reward function and set it to be proportional to the distance between the position of a placed block and its target position (since the RL algorithm operates on a one-block basis). The model of a policy combines learned embeddings of the image, target 3d array (where heights are interpreted as a channel), inventory and compass information in a single features vector that is passed to LSTM. We train the model for 2.5 billion environment steps using APPO. The performance (measured in a subtask success rate) plateaus after roughly 500 million environment steps (10-12 hours), but actual learning happens after since the task success rate starts growing exactly after that. For training, we used two Titan RTX GPUs used for rendering and 50 workers for sampling. We observed the same per env step sample efficiency when trained with one renderer GPU and 16 workers (with lower wall time efficiency). Note that we believe a successful solution should not necessarily modify the RL agent. The baseline has a lot of moving parts such as NLP model and a heuristic blocks iterator.


# Installation

Checkout this code repository. Make sure you have [git-lfs](https://git-lfs.github.com/) installed first, then do

```bash
git clone git@gitlab.aicrowd.com:aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline.git
```

Let's setup a dedicated Python environment for running the baseline.

```bash
conda create -y -n iglu_mhb python=3.9
conda activate iglu_mhb
```

This baseline version uses the latest version of [IGLU gridworld](https://github.com/iglu-contest/gridworld) and a specific version of [sample-factor](https://github.com/iglu-contest/sample-factory). Also, while this baseline has been tested with a specific version or PyTorch, more recent ones might work.

```bash
pip install git+https://github.com/iglu-contest/gridworld.git@master
pip install git+https://github.com/iglu-contest/sample-factory.git@1.121.3.iglu
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

All dependencies can be installed at once by running

```bash
pip install -r requirements.txt
```

# Training

## Training NLP T5 Model
Just run ```autoregressive_history.py``` from ``nlp_training`` directory:
```bash
python nlp_training/autoregressive_history.py
```
## Training RL APPO Agent
Just run ```train.py``` from ``rl_training`` directory with config_path:
```bash
python mhb_training/main.py --config_path iglu_baseline.yaml
```

# Evaluation

## Local Evaluation

To evaluate the NLP model run (the resulting plots will be saved in ``nlp-evaluation-plots``):

```bash
python local_nlp_evaluation.py
```

To evaluate the whole pipeline run (the resulting video will be saved in ``videos`` folder):

```bash
python local_evaluation.py
```

## AICrowd Evaluation

To submit your code to AICrowd for evaluation, you can use the `submit.sh` script which uses `aicrowd-cli` under-the-hood to make the submission process easier.

```bash
pip install -U aicrowd-cli
aicrowd login
```
Then,

```bash
./submit.sh <unique-submission-name>
```

More information about submitting to AICrowd can be found [here](https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-task-starter-kit/-/blob/master/docs/submission.md).

# Code Structure

The code for NLP model is located at `agents/mhb_baseline/nlp_model`.
The heuristic for blocks iteration is located at `wrappers/target_generator.py` (e.g. `target_to_subtasks` function, it implements the main algorithm for splitting the goal into subtasks).
In  `mhb_training/wrappers/multitask` you can find `TargetGenerator` and `SubtaskGenerator` classes. The first class makes a full-figure target using the `RandomFigure` generator or `DatasetFigure` generator.
The second class makes subtasks for the environment. The training config for the RL agent is in `iglu_baseline.yaml`. Most of the training code is implemented in the [sample-factory](https://github.com/iglu-contest/sample-factory) package. Most of the training curriculum is implemented via wrappers under `mhb_training/wrappers/` directory.

