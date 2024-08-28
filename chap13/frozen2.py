import gym
import numpy as np

env = gym.make('FrozenLake-v1', render_mode="human",  is_slippery=False)

discount_factor = 0.95
epsilon = 0.5
epsilon_decay_factor = 0.999
learning_rate = 0.8
num_episodes = 30000

q_table = np.zeros([env.observation_space.n, env.action_space.n])

for i in range(num_episodes):
    state = env.reset()
    epsilon *= epsilon_decay_factor	# 입실론: 탐사와 활용 비율 결정
    done = False

    while not done:
        if np.random.random() < epsilon:		# 난수가 입실론보다 작으면 탐사
            action = env.action_space.sample()	# 랜덤 액션
        else:					# 난수가 입실론보다 작으면 활용
            action = np.argmax(q_table[state, :])	# Q 테이블에서 가장 큰 값

        new_state, reward, done, _, _ = env.step(action)
	# 새로 얻은 정보로 Q-테이블 갱신
        q_table[state, action] +=   learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
        state = new_state
        if i==(num_episodes-1):
            env.render()