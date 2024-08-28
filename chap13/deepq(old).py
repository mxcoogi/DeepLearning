import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense

env = gym.make('FrozenLake-v0', is_slippery=False)

discount_factor = 0.95
epsilon = 0.5
epsilon_decay_factor = 0.999
num_episodes=500

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, env.observation_space.n)))
model.add(Dense(20, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

def one_hot(state):
    state_m=np.zeros((1, env.observation_space.n))
    state_m[0][state]=1
    return state_m

for i in range(num_episodes):		# 에피소드만큼 반복
    state = env.reset()		# 환경 초기화
    epsilon *= epsilon_decay_factor		# 입실론을 점점 작게 만든다. 
    done = False			# 게임 종료 여부	

    while not done:			# 게임이 종료되지 않았으면
        if np.random.random() < epsilon:	# 입실론보다 난수가 작으면 
            action = env.action_space.sample()	# 액션을 랜덤하게 선택
        else:
            action = np.argmax(model.predict(one_hot(state)))	# 가장 큰 Q 값 액션

	 # 게임을 한 단계 진행한다. 
        new_state, reward, done, _ = env.step(action)	# 게임 단계 진행

  	 # ① 목표값을 계산한다. 
        target = reward + discount_factor * np.max(model.predict(one_hot(new_state)))

  	 # ② 현재 상태를 계산한다.
        target_vector = model.predict(one_hot(state))[0]
        target_vector[action] = target

  	 # ③ 학습을 수행한다. 
        model.fit( one_hot(state), target_vector.reshape(-1, env.action_space.n), epochs=1, verbose=0)

  	 # 상태를 다음 상태로 바꾼다. 
        state = new_state
        print(i)
	 # ④ 마지막 상태만 화면에 표시한다. 
        if i==(num_episodes-1):
            env.render()