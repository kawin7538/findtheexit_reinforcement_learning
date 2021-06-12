import gym
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense

env = gym.make('FrozenLake-v0')

discount_factor = 0.95
eps = 0.5
eps_decay_factor = 0.999
learning_rate = 0.8
num_episodes = 500

# q_table=np.zeros([env.observation_space.n,env.action_space.n])
# # q_table=np.zeros([2,3])

# for i in range(num_episodes):
#     print("="*20)
#     print(" "*10+f"EPISODE {i+1}")
#     print("="*20)
#     state = env.reset()
#     eps*=eps_decay_factor
#     done=False
#     print(state)
#     while not done:
#         env.render()
#         # print(q_table)
#         if np.random.random() < eps or np.sum(q_table[state,:])==0:
#             action=np.random.randint(0,env.action_space.n)
#             # action=np.random.randint(0,3)
#         else:
#             action=np.argmax(q_table[state,:])
#         new_state, reward, done, info = env.step(action)
#         q_table[state,action] += (reward + learning_rate*(discount_factor*np.max(q_table[new_state,:])-q_table[state,action]))
#         state=new_state
#     env.render()
# env.close()
# print(q_table)

model=Sequential()
model.add(InputLayer(batch_input_shape=(1,env.observation_space.n)))
model.add(Dense(20,activation='relu'))
model.add(Dense(env.action_space.n,activation='linear'))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])

model.summary()

for i in range(num_episodes):
    print("="*20)
    print(" "*10+f"EPISODE {i+1}")
    print("="*20)
    state = env.reset()
    eps *= eps_decay_factor
    done = False
    while not done:
        env.render()
        if np.random.random() < eps:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(
              model.predict(np.identity(env.observation_space.n)[state:state + 1]))
        new_state, reward, done, _ = env.step(action)
        target = reward + discount_factor * np.max(model.predict(np.identity(env.observation_space.n)[new_state:new_state + 1]))
        target_vector = model.predict(np.identity(env.observation_space.n)[state:state + 1])[0]
        target_vector[action] = target
        model.fit(np.identity(env.observation_space.n)[state:state + 1], target_vector.reshape(-1, env.action_space.n), epochs=1, verbose=0)
        state = new_state
    env.render()

env.close()