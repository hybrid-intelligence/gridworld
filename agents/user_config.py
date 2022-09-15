
from agents.random_agent import RandomBuilderAgent
from agents.color_correct_random_agent import RandomBuilderAgent
from agents.mhb_baseline.agent import MultitaskHierarchicalAgent
###################################################################
#####        Specify your agent and other configs here        #####
###################################################################

class UserConfig:
    SingleAgent = MultitaskHierarchicalAgent

    # Only select between 'walking' and 'flying' (Invalid ones will default to walking)
    ActionSpaceName = 'walking'

    # Check agents/user_config_vector_agent.py if you want to use multiple environments in parallel
    # to process a batch of observations at once
    
