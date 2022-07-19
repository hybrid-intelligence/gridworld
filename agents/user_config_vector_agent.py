from agents.vector_agent import DummyVectorAgent

### Change this config if you want to use a batch of observations
### Set the number of parallel environments you want to use
### Change the code in the DummyVectorAgent class or implement a new class

# Explanation provided in agents/README.md 

class UserConfigVectorAgent:
    NumParallelEnvs = 1
    VectorAgent = DummyVectorAgent