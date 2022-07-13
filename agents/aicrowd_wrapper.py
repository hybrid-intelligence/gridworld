from gym.spaces import Discrete

from agents.user_config import UserConfig

class AIcrowdAgent:
    def __init__(self):
        self.num_parallel_envs = UserConfig.NumParallelEnvs
        self.agent = UserConfig.VectorAgent(UserConfig.NumParallelEnvs)
    
    def get_env_config(self):
        return {"num_parallel_envs": self.num_parallel_envs}
    
    def register_reset(self, observation):
        """Only called on first reset"""
        # self.agent.set_action_space(observation["action_space"])
        self.agent.set_action_space(Discrete(18))
        return self.compute_action(observation)

    def compute_action(self, observation):
        """Get batch of observations return actions"""
        return self.agent.act(*observation,)