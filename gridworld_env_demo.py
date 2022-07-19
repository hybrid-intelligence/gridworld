import gym
import gridworld
from local_evaluation import TaskGenerator, create_single_env

action_space_name = 'walking'
env = create_single_env(render=False, action_space_name=action_space_name)

taskgen = TaskGenerator(limit_tasks=None)
task_id, task = taskgen.get_next_task()
env.set_task(task)

print("Action space name:", action_space_name)
print()
print("Action Space:", env.action_space)
print()

action_space_name = 'flying'
env = create_single_env(render=False, action_space_name=action_space_name)

print("Action space name:", action_space_name)
print()
print("Action Space:", env.action_space)
