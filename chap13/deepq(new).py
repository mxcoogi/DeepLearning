import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from keras.optimizers import Adam

env = gym.make("FrozenLake-v1", is_slippery=False)

discount_factor = 0.9		# 할인률
epsilon = 1.0			# 입실론(탐사와 활용 비율)
num_episodes=4000		# 에피소드 수
max_steps=100			# 한 에피소드 당 최대 스텝
state_size = env.observation_space.n	# 상태의 수(16)
action_size = env.action_space.n	# 액션의 수(4)
learning_rate=0.01			# 신경망의 학습률

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, env.observation_space.n)))
model.add(Dense(20, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

memory = deque(maxlen=2500)	# 리플레이 버퍼
batch_size=32			# 리플레이 버퍼에서 한번에 꺼내는 개수

def one_hot(state):
    state_m=np.zeros((1, env.observation_space.n))
    state_m[0][state]=1
    return state_m

def replay(batch_size):
        global epsilon
        minibatch=random.sample(memory, batch_size)	# 배치 크기만큼 꺼낸다. 
        for new_state, reward, done, state, action in minibatch:
            target = reward
            # ① 목표값을 계산한다. 
            if not done:
                target = reward + discount_factor * np.max(model.predict(one_hot(new_state), verbose=0))
	     # ② 현재 상태를 계산한다.
            target_vector = model.predict(one_hot(state), verbose=0)
            target_vector[0][action]= target
            # ③ 학습을 수행한다. 
            model.fit(one_hot(state), target_vector, epochs=1, verbose=0)

        # 입실론을 수정한다. 
        if epsilon > 0.01:
            epsilon = 0.99 * np.exp(-0.0005*episode) + 0.01

for episode in range(num_episodes):		# 에피소드만큼 반복
    state, _ = env.reset()		# 환경 초기화
    done = False			# 게임 종료 여부	
    print(f"episode={episode} epsilon={epsilon}")

    for i in range(max_steps):
        if np.random.random() < epsilon:	# 입실론보다 난수가 작으면 
            action = env.action_space.sample()         # 액션을 랜덤하게 선택
        else:
            action = np.argmax(model.predict(one_hot(state), verbose=0))	# 가장 큰 Q 값 액션
        new_state, reward, done, _ , _= env.step(action)	# 게임 단계 진행
        memory.append((new_state, reward, done, state, action)) # 리플레이 버퍼에 저장
        state = new_state	# 새로운 상태로 바꾼다. 
        if done:
            print(f'에피소드 번호: {episode}/{num_episodes} 스텝: {i}  보상값 {reward}')
            break

    if len(memory)> batch_size :	# 어느 정도 리플레이 버퍼에 기록이 쌓이면
        replay(batch_size)		# 이때 학습이 이루어진다. 
