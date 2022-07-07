from gym.spaces import Discrete
from agents.user_config import UserConfig

class AIcrowdAgent:
    def __init__(self):
        self.num_parallel_envs = UserConfig.NumParallelEnvs
        self.vector_agent = UserConfig.VectorAgent(UserConfig.NumParallelEnvs)
    
    def get_env_config(self):
        return {"num_parallel_envs": self.num_parallel_envs}
    
    def _batch_check(self, observations):
        assert len(observations) == self.num_parallel_envs, \
            "Observations batch different from number of parallel envs"
    
    def register_reset(self, observation):
        """Only called on first reset"""
        # self.agent.set_action_space(observation["action_space"])
        self.vector_agent.set_action_space(Discrete(18))
        return self.compute_action(observation)

    def compute_action(self, observation):
        """Get batch of observations return actions"""
        observations, rewards, dones, infos = observation
        self._batch_check(observations)
        return self.vector_agent.act(observations, rewards, dones, infos)