import random
import numpy as np
from collections import deque
import tensorflow as tf

tf.keras.utils.disable_interactive_logging()


class DQNAgent:
    def __init__(
        self,
        actions=[
            (0, 1, 0),  # acceleration
            (-1, 0.2, 0),
            (0, 0, 0.3),  # break
            (1, 0.2, 0),
            (-1, 0.8, 0),
            (1, 0.8, 0),
        ],
        frame_stack_num=3,
        memory_size=5000,
        gamma=0.95,  # discount rate
        epsilon=1.0,  # exploration rate
        epsilon_min=0.1,
        epsilon_decay=0.9999,
        learning_rate=0.001,
    ):
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.actions = actions
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model(actions, learning_rate, frame_stack_num)
        self.target_model = self._build_model(actions, learning_rate, frame_stack_num)
        self.update_target_model()

    def _build_model(self, actions, learning_rate, frame_stack_num):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=5,
                    kernel_size=(6, 6),
                    strides=3,
                    activation="relu",
                    input_shape=(48, 48, frame_stack_num),
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2156, activation="relu"),
                tf.keras.layers.Dense(len(actions), activation=None),
            ]
        )

        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        )

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, self.actions.index(action), reward, next_state, done)
        )

    def do_action(self, state):
        if np.random.rand() > self.epsilon:
            act_values = self.model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randrange(len(self.actions))
        return self.actions[action_index]

    def replay(
        self,
        batch_size,
    ):
        minibatch = np.array(random.sample(self.memory, batch_size), dtype=object)
        states = np.array(minibatch[:, 0].tolist())
        actions = minibatch[:, 1].astype(int)
        rewards = minibatch[:, 2]
        next_states = np.array(minibatch[:, 3].tolist())
        dones = minibatch[:, 4]

        targets = self.model.predict(states)
        targets[np.arange(len(targets)), actions] = rewards + self.gamma * np.amax(
            self.target_model.predict(next_states), axis=1
        ) * (1 - dones)
        self.model.fit(
            states,
            targets,
            epochs=1,
            verbose=1,
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)
