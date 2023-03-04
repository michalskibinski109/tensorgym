import gym
import matplotlib.pyplot as plt
import numpy as np
import gym


# env = gym.make("CliffWalking-v0", render_mode="human")


# env.reset()
# truncated, terminated = False, False
# while not truncated and not terminated:
#     env.render()
#     move = env.action_space.sample()
#     observation, reward, truncated, terminated, info = env.step(move)
# env.close()


# q_table = np.zeros([env.observation_space.n, env.action_space.n])


# """Training the agent"""
# import random

# # Hyperparameters
# alpha = 0.2
# gamma = 0.6
# epsilon = 0.1
# # For plotting metrics
# all_epochs = []
# all_penalties = []
# for i in range(1, 30):
#     state = env.reset()[0]
#     epochs, penalties, reward, = (
#         0,
#         0,
#         0,
#     )
#     truncated, terminated = False, False
#     while not truncated:
#         if random.uniform(0, 1) < epsilon:
#             action = env.action_space.sample()  # Explore action space
#         else:
#             action = np.argmax(q_table[state])  # Exploit  learned values
#         (
#             next_state,
#             reward,
#             truncated,
#             terminated,
#             info,
#         ) = env.step(action)
#         # print(reward)
#         old_value = q_table[state, action]
#         next_max = np.max(q_table[next_state])

#         new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
#         q_table[state, action] = new_value
#         if reward == -100:
#             penalties += 1
#         state = next_state
#     epochs += 1

#     print(f"Episode: {i} reward: {reward} epochs: {epochs} penalties: {penalties}")
# print("Training finished.\n")
