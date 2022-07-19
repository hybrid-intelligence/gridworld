import numpy as np
from gym.spaces import Discrete, Dict, Box
from agents.user_config import UserConfig
from agents.user_config_vector_agent import UserConfigVectorAgent

def get_gridworld_action_space(action_space_name):
    """ Sets the action space same as evaluator - Do not change this """
    assert isinstance(action_space_name, str)
    if action_space_name.lower() == 'walking':
        return Discrete(18)
    elif action_space_name.lower() == 'flying':
        return Dict({
                'movement': Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                'camera': Box(low=-5, high=5, shape=(2,), dtype=np.float32),
                'inventory': Discrete(7),
                'placement': Discrete(3),
            })
    else:
        raise NotImplementedError("action space name should be walking or flying")

class AIcrowdAgent:
    def __init__(self):
        self.action_space_name = UserConfig.ActionSpaceName
        self.num_parallel_envs = UserConfigVectorAgent.NumParallelEnvs
        self.agent = UserConfigVectorAgent.VectorAgent(self.num_parallel_envs)
    
    def get_env_config(self):
        return {"num_parallel_envs": self.num_parallel_envs,
                "action_space_name": self.action_space_name}
    
    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def register_reset(self, observation):
        """Only called on first reset"""
        action_space = get_gridworld_action_space(self.action_space_name)
        self.agent.set_action_space(action_space)
        return self.compute_action(observation)

    def compute_action(self, observation):
        """Get batch of observations return actions"""
        return self.agent.act(*observation,)