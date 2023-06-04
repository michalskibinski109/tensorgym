import matplotlib.pyplot as plt
import json
import numpy as np

file = "training_data.json"
stats = json.load(open(file, "r"))


def display_training_stats(stats):
    """
        "0": {
        "scores": 97,
        "total_reward": -0.38442622950815425,
        "epsilon": 1
    },
        "1": {
        "scores": 17,
        "total_reward": -0.3342622950815425,
        "epsilon": .9
    },

    """
    episodes = []
    scores = []
    total_rewards = []
    epsilons = []

    for episode, data in stats.items():
        episodes.append(episode)
        scores.append(data["scores"])
        total_rewards.append(data["total_reward"])
        epsilons.append(data["epsilon"])
    # Compute the 50-moving average
    window = 50
    scores_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
    total_rewards_avg = np.convolve(
        total_rewards, np.ones(window) / window, mode="valid"
    )

    # Create a figure and axis objects
    fig, ax = plt.subplots(2, figsize=(10, 6))

    # Plotting real values
    ax[0].plot(episodes, scores, label="Scores")
    ax[1].plot(episodes, total_rewards, label="Total Rewards")
    # ax[1].plot(episodes, epsilons, label="Epsilon Values")

    # Plotting 50-moving average
    ax[0].plot(episodes[window - 1 :], scores_avg, label="Moving Average")
    ax[1].plot(episodes[window - 1 :], total_rewards_avg, label="Moving Average")

    # Plotting all stats without moving average
    # ax[0].plot(episodes, scores, label="Scores")
    # ax[1].plot(episodes, total_rewards, label="Total Rewards")
    # ax[2].plot(episodes, epsilons, label="Epsilon Values")

    # Add titles
    ax[0].set_title("Scores over Episodes")
    ax[1].set_title("Total Rewards over Episodes")

    # Add legend
    ax[0].legend()
    ax[1].legend()

    # set the x-stick labels
    ax[0].set_xticks(range(0, len(episodes), 10))
    ax[1].set_xticks(range(0, len(episodes), 10))

    # Add x and y axis labels
    ax[1].set_xlabel("Episodes")
    ax[0].set_ylabel("Scores")
    ax[1].set_ylabel("Total Rewards")

    plt.tight_layout()  # Improves spacing between subplots
    plt.show()


# Call the function to display and plot the training stats
display_training_stats(stats)
