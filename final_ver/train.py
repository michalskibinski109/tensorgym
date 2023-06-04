import argparse
import json
import gymnasium as gym
from collections import deque
from agent import DQNAgent
from utils import process_image, transpose_frames_stack
import tensorflow as tf
from tqdm import tqdm
from collections import Counter

START_EPISODE = 241
END_EPISODE = 600
SKIP_FRAMES = 2
TRAINING_BATCH_SIZE = 128
SAVE_TRAINING_FREQUENCY = 40
UPDATE_TARGET_MODEL_FREQUENCY = 5


if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    #### FROM ONE POINT TO ANOTHER ####
    agent = DQNAgent(epsilon=1)

    agent.load("trained_model_240.h5")
    agent.epsilon = 0.1
    # STARTING_EPISODE = 60

    training_data = {}
    for e in range(START_EPISODE, END_EPISODE + 1):
        init_state, info = env.reset()
        init_state = process_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state] * 3, 3)
        frame = 1
        done = False

        bar = tqdm(range(1000))
        while True:
            current_state_frame_stack = transpose_frames_stack(state_frame_stack_queue)
            action = agent.do_action(current_state_frame_stack)

            reward = 0
            for _ in range(SKIP_FRAMES):
                next_state, r, done, trunkaced, _ = env.step(action)
                reward += r
                if done or trunkaced:
                    break
            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = (
                negative_reward_counter + 1 if frame > 100 and reward < 0 else 0
            )

            # Extra bonus for the model if it uses full gas
            if action[1] > 0.6 and action[2] == 0:
                reward *= 1.5

            total_reward += reward
            next_state = process_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = transpose_frames_stack(state_frame_stack_queue)

            agent.update_memory(
                current_state_frame_stack,
                action,
                reward,
                next_state_frame_stack,
                (done or trunkaced),
            )
            bar.update(1)
            bar.set_description(
                f"Episode: {e}, reward {reward:.2f} frames : {frame}, Total Rewards: {total_reward:.2f}, E: {agent.epsilon:.2f}"
            )
            if done or trunkaced or negative_reward_counter >= 25 or total_reward < 0:
                training_data[e] = {
                    "scores": frame,
                    "total_reward": total_reward,
                    "epsilon": agent.epsilon,
                    "actions": dict(
                        Counter([a for _, a, _, _, _ in list(agent.memory)[-frame:]])
                    ),
                }
                bar.set_description(
                    f"Episode: {e}, Total Rewards: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, memory: {len(agent.memory)} "
                )
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
            frame += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            with open(f"training_data.json", "w") as f:
                json.dump(training_data, f)
            agent.save(f"trained_model_{e}.h5")

    env.close()
