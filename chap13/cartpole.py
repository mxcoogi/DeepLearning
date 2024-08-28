import pygame
import gym

#env = gym.make("CartPole-v1")
env = gym.make("CartPole-v1", render_mode='human')
env.reset()		# (2)

for _ in range(1000):			# (3)	
  env.render()				# (4)
  action = env.action_space.sample() 		# (5)
  obs, reward, done, truncated, info = env.step(action)

  if done:
    env.reset()		# (7)
env.close()
