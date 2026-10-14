import numpy as np
import pickle
import matplotlib.pyplot as plt
from env import RandomHazardEnv
from agent import QLearningAgent, SARSAAgent

def train_q_learning(env, agent, episodes=5000):
    returns = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        agent.decay_epsilon()
        returns.append(total_reward)
    return returns

def train_sarsa(env, agent, episodes=5000):
    returns = []
    for ep in range(episodes):
        state, _ = env.reset()
        action = agent.select_action(state)
        done = False
        total_reward = 0
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = agent.select_action(next_state)
            
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            total_reward += reward
            
        agent.decay_epsilon()
        returns.append(total_reward)
    return returns

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

if __name__ == "__main__":
    EPISODES = 6000
    
    # Cấu hình Setting 3: Hazard Chasing + Smoke Bomb
    env = RandomHazardEnv(grid_size=6, hazard_type="chasing", num_hazards=1, has_smoke_item=True)
    
    print("--- Huấn luyện Q-Learning ---")
    q_agent = QLearningAgent()
    q_returns = train_q_learning(env, q_agent, episodes=EPISODES)
    
    print("--- Huấn luyện SARSA ---")
    sarsa_agent = SARSAAgent()
    sarsa_returns = train_sarsa(env, sarsa_agent, episodes=EPISODES)
    
    # Lưu Q-table để làm demo
    with open("trained_model.pkl", "wb") as f:
        pickle.dump({"q_table": q_agent.q_table, "sarsa_table": sarsa_agent.q_table}, f)
        
    # Vẽ đồ thị so sánh
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(q_returns, 100), label="Q-Learning", color="blue")
    plt.plot(moving_average(sarsa_returns, 100), label="SARSA", color="red")
    plt.title("So sánh đường cong học tập (Setting 3: Chasing + Smoke Bomb)")
    plt.xlabel("Episode (Moving Average 100)")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve_comparison.png")
    plt.show()
    print("Đã lưu đồ thị so sánh vào file 'learning_curve_comparison.png'")