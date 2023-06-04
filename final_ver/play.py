import gymnasium as gym
from agent import DQNAgent
from utils import process_image, transpose_frames_stack
import cv2
from matplotlib import pyplot as plt
import imageio
from collections import deque


EPISODES = 6
GIF_FILENAME = "agent_view.gif"

rewards = []


def show_how_agent_sees_picture():
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    init_state, _ = env.reset()
    init_state_processed = process_image(init_state)
    done, truncated = False, False
    cv2.namedWindow("Agent View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)

    cv2.imshow("Agent View", init_state_processed)
    cv2.imshow("Original Image", init_state)
    cv2.waitKey(1)
    frames = []
    while (not done or truncated) and len(frames) < 700:
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        next_state_processed = process_image(next_state)  # reutrns gryscale image 48x48
        rewards.append(reward)
        # next_state = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
        cv2.imshow("Agent View", next_state_processed)
        cv2.imshow("Original Image", next_state)
        cv2.waitKey(1)
        frames.append(next_state_processed * 255)
    cv2.destroyAllWindows()
    # get 1/3 of the frames\
    imageio.mimsave(GIF_FILENAME, frames[400:700:3], duration=0.005)
    return rewards


def play_by_model():
    train_model = "trained_model_240.h5"

    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    env = gym.wrappers.ResizeObservation(env, (256, 256))
    agent = DQNAgent(
        epsilon=0
    )  # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(train_model)
    bad_frames = 0
    frames = []
    for e in range(EPISODES):
        init_state, _ = env.reset()
        init_state = process_image(init_state)
        state_frame_stack_queue = deque([init_state] * 3, 3)

        while True:
            current_state_frame_stack = transpose_frames_stack(state_frame_stack_queue)
            action = agent.do_action(current_state_frame_stack)
            next_state, reward, done, trunkaced, _ = env.step(action)
            frames.append()
            next_state = process_image(next_state)
            state_frame_stack_queue.append(next_state)
            if reward < 0:
                bad_frames += 1
            else:
                bad_frames = 0

            if done or trunkaced or bad_frames > 50:
                imageio.mimsave(f"agent_{e}.gif", frames, duration=(1000 * 1 / 120))
                frames = []
                break


if __name__ == "__main__":
    # show_how_agent_sees_picture()
    play_by_model()
