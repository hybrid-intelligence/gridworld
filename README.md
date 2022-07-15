![IGLU Banner](https://user-images.githubusercontent.com/660004/179000978-29cf4462-4d2b-4623-8418-157449322fda.png)

# **[NeurIPS 2022 - IGLU Challenge](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge)** - Starter Kit


This repository is the IGLU Challenge **Starter kit**! It contains:
* **Instructions** for setting up your codebase to make submissions easy.
* **Baselines** for quickly getting started training your agent.
* **Documentation** for how to submit your model to the leaderboard.

Quick Links:

* [The IGLU Challenge - Competition Page](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge)
* [The IGLU Challenge - Slack Workspace](https://join.slack.com/t/igluorg/shared_invite/zt-zzlc1qpy-X6JBgRtwx1w_CBqOV5~jaA&sa=D&sntz=1&usg=AOvVaw33cSaYXeinlMWYC6bGIe33)
* [The IGLU Challenge - Starter Kit](https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022)

## Quick Start

TODO

With Docker and x1 GPU
```bash
# 1. CLONE THE REPO AND DOWNLOAD BASELINE MODELS
git clone http://gitlab.aicrowd.com/iglu/neurips-2022-the-iglu-challenge.git \
    && cd neurips-2022-the-iglu-challenge

# 2. START THE DOCKER IMAGE
...

# 3. TEST AN EXISTING SUBMISSION 
python test_submission.py      # Tests ./saved_models/TODO

# 3. TRAIN YOUR OWN
...
```


# Table of Contents
1. [Intro to IGLU Gridworld and the IGLU Challenge](#intro-to-iglu-gridworld-and-the-iglu-challenge)
2. [Setting up your codebase](#setting-up-your-codebase)
3. [Baselines](#baselines)
4. [How to test and debug locally](#how-to-test-and-debug-locally)
5. [How to submit](#how-to-submit)

# Intro to IGLU Gridworld and the IGLU Challenge

Your goal of this challenge is to **build interactive agents** that learn to solve a task while provided with **grounded natural language instructions** in a **collaborative environment**.

TODO

#### A high level description of the Challenge Procedure:
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge).
2. **Clone** this repo and start developing your solution.
3. **Train** your models on IGLU, and ensure run.sh will generate rollouts.
4. **Submit** your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com)
for evaluation (full instructions below). The automated evaluation setup
will evaluate the submissions against the IGLU Gridworld environment for a fixed 
number of rollouts to compute and report the metrics on the leaderboard
of the competition.

## IGLU Reinforcement Learning Environment

IGLU Gridworld is a fast and scalable reinforcement learning environment. The env represents an embodied agent with an ability to navigate, place, and break blocks of six different colors.

### Action spaces:

Two action spaces are available in Gridworld:

**Walking actions** allow the agent to move with gravity enabled, jump, place, break, and select blocks from inventory. 

```python
env = gym.make('IGLUGridworld-v0', action_space='walking')
print(env.action_space) # Discrete(18)
```

The action space format is the following:

  * 0 - no-op
  * 1 - step forward
  * 2 - step backward
  * 3 - step left 
  * 4 - step right
  * 5 - jump
  * 6-11 - inventory select
  * 12 - move the camera left
  * 13 - move the camera right
  * 14 - move the camera up
  * 15 - move the camera down
  * 16 - break block 
  * 17 - place block

For each movement action, the agent steps for about 0.25 of one block. Camera movement changes each angle by 5 degrees. 

**Flying actions** allow the agent to fly freely within the building zone. Placement actions are the same and movement is specified by a continuous vector.

```python
env = gym.make('IGLUGridworld-v0', action_space='flying')
```

Action space format:

```python
Dict(
  movement: Box(low=-1, high=1, shape=(3,)),
  camera: Box(low=-5, high=5, shape=(2,)),
  inventory: Discrete(7),
  placement: Discrete(3)
)
```

### Observation space

Observation space format:

```python
Dict(
  inventory: Box(low=0, high=20, shape=(6,)),
  compass: Box(low=-180, high=180, shape=(1,)),
  dialog: String(),
  pov: Box(low=0, high=255, shape=(64, 64, 3))
)
```

Here, `inventory` indicates the total blocks available to the agent (per color).
The `compass` component shows the angle between the agent's yaw angle and the North direction.
The `dialog` value is the full previous dialog and the most recent instruction to execute.
Finally, `pov` is an ego-centric image of the agent's observation of the world.
Note that **this space will be used during the evaluation.**
However, it is possible to access other fields of the environment, for example, during training.
The `vector_state=True` passed as keyword argument in `gym.make` will return, in addition to previous fields,

```
agentPos: Box(low=[-8, -2, -8, -90, 0], high=[8, 12, 8, 90, 360], shape=(5,)),
grid: Box(low=-1, high=7, shape=(9, 11, 11))
```

It is also possible to make a target grid a part of the observation space. To do that, pass `target_in_obs=True` to `gym.make`. This will add another key to the observation space with the same structure as the `grid` component. The name of a new component is `target_grid`. This part of the space remains fixed within an episode.

### Reward calculation

Each step, the reward is calculated based on the similarity between the so far built grid and the target grid. The reward is determined regardless of global spatial position of currently placed blocks, it only takes into account how much the built blocks are similar to the target structure. To make it possible, at each step we calculate the intersection between the built and the target structures for each spatial translation within the horizontal plane and rotation around the vertical axis. Then we take the maximal intersection value among all translation and rotations. To calculate the reward, we compare the maximal intersection size from the current step with the one from the previous step. We reward the agent with `2` for the increase of the maximal intersection size, with `-2` for the decrease of the maximal intersection size, and with `1`/`-1` for removing/placing a block without a change of the maximal intersection size. A visual example is shown below.

<img src="./assets/intersections.png" width="256">

Specifically, we run the code that is equivalent to the following one:

```python
def maximal_intersection(grid, target_grid):
  """
  Args:
    grid (np.ndarray[Y, X, Z]): numpy array snapshot of a built structure
    target_grid (np.ndarray[Y, X, Z]): numpy array snapshot of the target structure
  """
  maximum = 0
  # iterate over orthogonal rotations
  for i in range(4):
    # iterate over translations
    for dx in range(-X, X + 1):
      for dz in range(-Z, Z + 1):
        shifted_grid = translate(grid, dx, dz)
        intersection = np.sum( (shifted_grid == target) & (target != 0) )
        maximum = max(maximum, intersection)
    grid = rotate_y_axis(grid)
  return maximum
```

In practice, a more optimized version is used. The reward is then calculated based on the temporal difference between maximal intersection of the two consecutive grids. Formally, suppose `grids[t]` is a built structure at timestep `t`. The reward is then calculated as:

```python
def calc_reward(prev_grid, grid, target_grid, , right_scale=2, wrong_scale=1):
  prev_max_int = maximal_intersection(prev_grid, target_grid)
  max_int = maximal_intersection(grid, target_grid)
  diff = max_int - prev_max_int
  prev_grid_size = num_blocks(prev_grid)
  grid_size = num_blocks(grid)
  if diff == 0:
    return wrong_scale * np.sign(grid_size - prev_grid_size)
  else:
    return right_scale * np.sign(diff)
```

In other words, if a recently placed block strictly increases or decreases the maximal intersection, the reward is positive or negative and is equal to `+/-right_scale`. Otherwise, its absolute value is equal to `wrong_scale` and the sign is positive if a block was removed or negative if added.
Values `right_scale` and `wrong_scale` can be passed to `gym.make` as environment kwargs. Finally, the `maximal_intersection` includes heavy computations that slow down the environment. They can be simplified by disabling rotational/translational invariance at the cost of much more sparse reward. To do that, pass `invariant=False` to a corresponding `Task` object (see Dataset section for reference).


# Setting Up Your Codebase

AIcrowd provides great flexibility in the details of your submission!  
Find the answers to FAQs about submission structure below, followed by 
the guide for setting up this starter kit and linking it to the AIcrowd 
GitLab.

## FAQs

### How does submission work?

The submission entrypoint is a bash script `run.sh`, that runs in an environment defined by `Dockerfile`. When this script is 
called, aicrowd will expect you to generate all your rollouts in the 
allotted time, using `aicrowd_gym` in place of regular `gym`.  This means 
that AIcrowd can make sure everyone is running the same environment, 
and can keep score!

### What languages can I use?

Since the entrypoint is a bash script `run.sh`, you can call any arbitrary
code from this script.  However, to get you started, the environment is 
set up to generate rollouts in Python. 

The repo gives you a template placeholder to load your model 
(`agents/your_agent.py`), and a config to choose which agent to load 
(`submission_config.py`). You can then test a submission, adding all of 
AIcrowd’s timeouts on the environment, with `python test_submission.py`

### How do I specify my dependencies?

We accept submissions with custom runtimes, so you can choose your 
favorite! The configuration files typically include `requirements.txt` 
(pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the [RUNTIME.md](/docs/RUNTIME.md) file.

### What should my code structure look like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

TODO

```
.
├── aicrowd.json                  # Submission meta information - add tags for tracks here
├── apt.txt                       # Packages to be installed inside submission environment
├── requirements.txt              # Python packages to be installed with pip
├── rollout.py                    # This will run rollouts on a batched agent
├── test_submission.py            # Run this on your machine to get an estimated score
├── run.sh                        # Submission entrypoint
├── utilities                     # Helper scripts for setting up and submission 
│   └── submit.sh                 # script for easy submission of your code
├

```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** See [How do I actually make a submission?](#how-do-i-actually-make-a-submission) below for more details.


### How can I get going with an existing baseline?

The best current baseline is ... TODO

To then submit your saved model, simply set the `AGENT` in 
`submission config` to be `TODO`, and modify the 
`agent/TODO_agent.py` to point to your saved directory.

You can now test your saved model with `python test_submission.py`

### How can I get going with a completely new model?

Train your model as you like, and when you’re ready to submit, just adapt
`YourAgent` in `agents/your_agent.py` to load your model and take a `batched_step`.

Then just set your `AGENT` in `submission_config.py` to be this class 
and you are ready to test with `python test_submission.py`

### How do I actually make a submission?

First you need to fill in you `aicrowd.json`, to give AIcrowd some info so you can be scored.
The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "neurips-2022-the-iglu-challenge",
  "authors": ["your-aicrowd-username"],
  "description": "(optional) description about your awesome agent",
  "gpu": true
}
```

The submission is made by adding everything including the model to git,
tagging the submission with a git tag that starts with `submission-`, and 
pushing to AIcrowd's GitLab. The rest is done for you!

More details are available [here](/docs/SUBMISSION.md).

### Are there any hardware or time constraints?

TODO

Your submission will need to complete X rollouts in Y minutes. 

The machine where the submission will run will have following specifications:
* 1 NVIDIA T4 GPU
* 4 vCPUs
* 16 GB RAM


## Setting Up Details [No Docker]

1. **Add your SSH key** to AIcrowd GitLab

    You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:iglu/neurips-2022-the-iglu-challenge.git
    ```
    
3. **Verify you have dependencies** for the IGLU Gridworld Environment

    IGLU Gridworld requires `python>=3.7` to be installed and available both when building the
    package, and at runtime.
    
4. **Install** competition specific dependencies!

    We advise using a conda environment for this:
    ```bash
    # Optional: Create a conda env
    conda create -n iglu_challenge python=3.8
    conda activate iglu_challenge
    pip install -r requirements.txt
    ```

5. **Run rollouts** with a random agent with `python test_submission.py`.


## Setting Up Details [Docker]

With Docker, TODO...

## Testing IGLU framework:

Note that by default IGLU env runs in headless mode. To run headed do 

```sh
export IGLU_HEADLESS=0
```

Now, run the environment loop:

```python
import gym
import gridworld
from gridworld.tasks import DUMMY_TASK

# create vector based env. Rendering is enabled by default. 
# To turn it off, use render=False.
env = gym.make('IGLUGridworld-v0')

# It is mandatory to task the environemnt. 
# For dummy looping, you can use a DUMMY_TASK object:
env.set_task(DUMMY_TASK)
done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```

# Baselines

Although we are looking to supply this repository with baselines... TODO

# How to Test and Debug Locally

The best way to test your model is to run your submission locally.

You can do this naively by simply running  `python rollout.py` or you can simulate the extra timeout wrappers that AIcrowd will implement by using `python test_submission.py`. 

# How to Submit

More information on submissions can be found at our [SUBMISSION.md](/docs/SUBMISSION.md).

## Contributors

TODO

- [Dipam Chakraborty](https://www.aicrowd.com/participants/dipam)


# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/neurips-2022-the-iglu-challenge
- 🗣️ Discussion Forum: https://www.aicrowd.com/challenges/neurips-2022-the-iglu-challenge/discussion
- 🏆 Leaderboard: https://www.aicrowd.com/challenges/neurips-2022-the-iglu-challenge/leaderboards

**Best of Luck** 🎉 🎉
