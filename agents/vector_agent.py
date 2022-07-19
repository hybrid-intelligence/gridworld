from agents.user_config import UserConfig

class DummyVectorAgent:
    """ 
    The AIcrowd evalautor reqiures a Vector Agent
    Here is a dummy vector agent that loops over all the observations
    If your agent is fast enough, leave this vector agent as is
    """    
    def __init__(self, num_parallel_envs):
        """ Set the number of parallel envs, change this in user config as per your requirement"""
        self.num_parallel_envs = num_parallel_envs # Don't change this here, see user_config.py
        self.action_space = None # Don't change this, will be set by aicrowd wrapper

        Agent_cls = UserConfig.SingleAgent
        self.agents = [Agent_cls() for _ in range(self.num_parallel_envs)]
    
    def set_action_space(self, action_space):
        """ AIcrowd wrapper will call this with the appropriate action space """
        for agent in self.agents:
            agent.set_action_space(action_space)

    def act(self, observations, rewards, dones, infos):
        """Get batch of observations return actions"""
        actions = []
        early_terminations = []
       
        for agent, obs, rew, done, info in \
                zip(self.agents, observations, rewards, dones, infos):
            action, early_termination = agent.act( obs, rew, done, info)
            actions.append(action)
            early_terminations.append(early_termination)

         #### You can also process this as a batch call to a neural net
         ### Example: 
         ### batch_obs = torch.Tensor(obs)
         ### actions = batched_nn_agent(batch_obs)
         ### actions = actions.tolist()

        return actions, early_terminations
    