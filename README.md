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
