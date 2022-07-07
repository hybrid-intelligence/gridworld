class RandomBuilderAgent:
    """
    Processes observations from a single env
    """
    def __init__(self, action_space):
        self.actions_space = None

    def set_action_space(self, action_space):
        self.actions_space = action_space

    def act(self, observation, reward, done, info):
        return self.actions_space.sample()