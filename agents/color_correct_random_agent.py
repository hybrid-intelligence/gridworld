import numpy as np

class RandomBuilderAgent:
    """
    Processes observations from a single env.

    This simple baseline shows the simplest heuristic agent: it avoids placing 
    blocks of colors not mentioned in the instruction. Other actions are allowed and
    disctribution over them is uniform.
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
        text = observation['dialog']
        colors_to_hotbar = {
            'blue': 1,  # blue
            'green': 2,  # green
	        'red': 3,  # red
            'orange': 4,  # orange
            'purple': 5,  # purple
            'yellow': 6,  # yellow
        }
        hotbars_turned_off = []
        for color, hotbar in colors_to_hotbar.items():
            if color not in text:
                hotbars_turned_off.append(hotbar)
        hotbars_turned_off = set(hotbars_turned_off)
        while (action - 5) in hotbars_turned_off: # actions 6 to 11 are block placements
            action = self.actions_space.sample()
        # Terminate early with 1% chance
        user_termination = np.random.choice([True, False], p=[0.01, 0.99]) 
        return action, user_termination
