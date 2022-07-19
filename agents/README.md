## How to write your own agent?

We recommend that you place the code for all your agents in the `agents` directory (though it is not mandatory). You should implement the `act` function.

**Add your agent name in** `user_config.py`
  
See the example in `agents/random_agent.py`

## Parallel Environments

Since IGLU-Gridworld is super fast, you may want to run multiple envs in parallel and process a batch of observations at once. To set the number of parallel envs change `UserConfigVectorAgent.NumParallelEnvs` in `agents/user_config_vector_agent.py`

We provide a dummy vector agent that just loops over all the observations. You can change this class to process batch observations if you want.

Replace the code in `DummyVectorAgent` in `vector_agent.py` or implement your own vector agent and change the `UserConfigVectorAgent.VectorAgent` in `agents/user_config_vector_agent.py`

Here's and example (for illustration only) 
```
    batch_obs = torch.Tensor(obs)
    actions = batched_nn_agent(batch_obs)
    actions = actions.tolist()
```

Depending on your agent implementation and hardware requirements, you may be able to get much higher throughput by using parallel environments such that the GPU is fully utilized.

## What's used by the evaluator

The evaluator uses `AIcrowdAgent` from `aicrowd_wrapper.py` as its entrypoint. You do not need to edit this file. `local_evaluation.py` shows how the evaluator interacts with the agent.