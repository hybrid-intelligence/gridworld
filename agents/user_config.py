
from agents.random_agent import RandomBuilderAgent

###################################################################
#####        Specify your agent and other configs here        #####
###################################################################

class UserConfig:
    SingleAgent = RandomBuilderAgent

    # Only select between 'walking' and 'flying' (Invalid ones will default to walking)
    ActionSpaceName = 'flying'

    # Check agents/user_config_vector_agent.py if you want to use multiple environments in parallel
    # to process a batch of observations at once
    