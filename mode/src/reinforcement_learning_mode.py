"""
Reinforcement Learning Mode for Piksu Bird AI project
This file implements the 2D flying birds environment and reinforcement learning structure.
"""

import numpy as np
import random
import csv

NUM_EPISODES = 1000  # Increased number of episodes for better learning
GRAVITY = 0.5
LIFT = 1.0
MAX_VELOCITY = 5

class Environment:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.bird_position = None
        self.bird_velocity = None
        self.target_position = None
        self.obstacles = []
        self.bird_head_angle = 0
        self.is_standing = False

    def reset(self):
        # Reset bird to a random position with zero velocity
        self.bird_position = np.array([np.random.randint(0, self.width),
                                       np.random.randint(0, self.height // 2)])
        self.bird_velocity = np.array([0, 0])
        # Set target to a random position
        self.target_position = np.array([np.random.randint(0, self.width),
                                         np.random.randint(self.height // 2, self.height)])
        # Generate some random obstacles
        self.obstacles = [np.array([np.random.randint(0, self.width),
                                    np.random.randint(0, self.height)])
                          for _ in range(3)]  # 3 obstacles for now
        # Initialize new attributes
        self.bird_head_angle = 0
        self.is_standing = True if self.bird_position[1] == 0 else False
        return self._get_state()

    def step(self, action):
        # Apply action
        # 0: do nothing, 1: flap (move up), 2: move right, 3: move left, 4: turn head left, 5: turn head right
        if action == 1:
            self.bird_velocity[1] += LIFT
        elif action == 2:
            self.bird_velocity[0] = min(MAX_VELOCITY, self.bird_velocity[0] + 1)
        elif action == 3:
            self.bird_velocity[0] = max(-MAX_VELOCITY, self.bird_velocity[0] - 1)
        elif action == 4:  # Turn head left
            self.bird_head_angle = (self.bird_head_angle - 15) % 360
        elif action == 5:  # Turn head right
            self.bird_head_angle = (self.bird_head_angle + 15) % 360

        # Apply gravity
        self.bird_velocity[1] -= GRAVITY

        # Update position
        self.bird_position += self.bird_velocity

        # Constrain bird within boundaries
        self.bird_position[0] = np.clip(self.bird_position[0], 0, self.width - 1)
        self.bird_position[1] = np.clip(self.bird_position[1], 0, self.height - 1)

        # Update standing state
        self.is_standing = (self.bird_velocity[0] == 0 and self.bird_velocity[1] == 0 and self.bird_position[1] == 0)

        # Check if bird reached the target
        done = np.array_equal(self.bird_position.astype(int), self.target_position)

        # Calculate reward
        reward = self._calculate_reward()

        return self._get_state(), reward, done

    def _get_state(self):
        # Return the state as a flattened array
        return np.concatenate([self.bird_position, self.bird_velocity,
                               [self.bird_head_angle], [int(self.is_standing)],
                               self.target_position] + [obs for obs in self.obstacles])

    def _calculate_reward(self):
        # Calculate distance to target
        distance = np.linalg.norm(self.bird_position - self.target_position)

        # Check if bird hit an obstacle
        if any(np.array_equal(self.bird_position.astype(int), obs) for obs in self.obstacles):
            return -10  # Penalty for hitting an obstacle

        # Reward inversely proportional to distance
        return 10 if distance == 0 else 1 / distance

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.action_size)]
            return np.argmax(q_values)

    def learn(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in range(self.action_size)])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(tuple(state), action)] = new_q

    def decay_epsilon(self, episode, total_episodes):
        self.epsilon = max(0.01, self.epsilon * (1 - episode / total_episodes))

# Function to save the dataset
def save_dataset(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['State', 'Action', 'Reward', 'Next State', 'Done'])
        writer.writerows(data)

# Main loop for the reinforcement learning mode
if __name__ == "__main__":
    # Initialize environment and agent
    environment = Environment()
    state_size = 14  # 2 for position, 2 for velocity, 1 for head angle, 1 for standing, 2 for target, 2 * 3 for obstacles
    action_size = 6  # do nothing, flap, move right, move left, turn head left, turn head right
    agent = Agent(state_size, action_size)

    # Initialize dataset
    dataset = []

    # Run episodes for training
    for episode in range(NUM_EPISODES):
        state = environment.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = environment.step(action)
            dataset.append([state.tolist(), action, reward, next_state.tolist(), done])
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        print(f"Episode {episode}:")
        print(f"  Final State: {state.tolist()}")
        print(f"  Bird Position: ({state[0]:.2f}, {state[1]:.2f}), Velocity: ({state[2]:.2f}, {state[3]:.2f})")
        print(f"  Head Angle: {state[4]:.2f}, Standing: {'Yes' if state[5] else 'No'}")
        print(f"  Total Reward: {total_reward}")
        print(f"  Q-table size: {len(agent.q_table)}")
        print("---")

    print("Training completed.")
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample data point: {dataset[0]}")
    save_dataset(dataset, 'flying_birds_dataset.csv')
    print("Dataset saved to flying_birds_dataset.csv")