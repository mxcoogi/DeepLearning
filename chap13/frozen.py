import gym

env = gym.make("FrozenLake-v0", is_slippery=False)
observation = env.reset()

for _ in range(100):
  env.render()
  action = env.action_space.sample() 	# (1)
  observation, reward, done, info = env.step(action)  # (2)

  if done:
    observation = env.reset()
env.close()