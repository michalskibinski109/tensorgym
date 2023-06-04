import cv2
import numpy as np


def process_image(state):
    new_state = state.copy()
    new_state = new_state[:85, :]
    new_state = cv2.erode(new_state, np.ones((2, 2), np.uint8), iterations=4)
    # new_state = cv2.bitwise_not(new_state, new_state, mask=mask)
    new_state = cv2.cvtColor(new_state, cv2.COLOR_RGB2GRAY)
    new_state = cv2.Canny(new_state, 150, 150)
    new_state = cv2.resize(new_state, (48, 48))

    new_state = new_state.astype(float)
    new_state /= 255.0
    return new_state


def transpose_frames_stack(deque):
    frame_stack = np.array(deque)
    return np.transpose(frame_stack, (1, 2, 0))
