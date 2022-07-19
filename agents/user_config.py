
from agents.random_agent import RandomBuilderAgent

###################################################################
#####        Specify your agent and other configs here        #####
###################################################################

class UserConfig:
    SingleAgent = RandomBuilderAgent

    # Only select between 'walking' and 'flying' (Invalid ones will default to walking)
    ActionSpaceName = 'flying'
    