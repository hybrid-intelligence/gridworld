import numpy as np

class RandomBuilderAgent:
    """
    Processes observations from a single env
    """
    def __init__(self):
        self.actions_space = None

    def set_action_space(self, action_space):
        self.actions_space = action_space

    def act(self, observation, reward, done, info):
        """ 
            Input a single observation 
            Return action, user_termination
                user_termination - Return True to terminate the episode
        """
        action = self.actions_space.sample()
        # Terminate early with 1% chance
        user_termination = np.random.choice([True, False], p=[0.01, 0.99]) 
        return action, user_termination