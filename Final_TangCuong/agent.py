import numpy as np

class QLearningAgent:
    def __init__(self, action_space_size=5, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
        self.n_actions = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        
    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.get_q_values(state))
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.get_q_values(state)[action]
        next_max_q = np.max(self.get_q_values(next_state)) if not done else 0.0
        
        # Cập nhật Off-policy: max Q(s', a')
        td_target = reward + self.gamma * next_max_q
        self.q_table[state][action] += self.alpha * (td_target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class SARSAAgent:
    def __init__(self, action_space_size=5, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
        self.n_actions = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.get_q_values(state))

    def update(self, state, action, reward, next_state, next_action, done):
        current_q = self.get_q_values(state)[action]
        next_q = self.get_q_values(next_state)[next_action] if not done else 0.0
        
        # Cập nhật On-policy: Q(s', a') theo hành động tiếp theo
        td_target = reward + self.gamma * next_q
        self.q_table[state][action] += self.alpha * (td_target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)