import time
import random
import numpy as np
import matplotlib.pyplot as plt

class RandomHazardEnvDemo:
    def __init__(self, grid_size=6, hazard_type="chasing", has_smoke_item=True):
        self.grid_size = grid_size
        self.hazard_type = hazard_type
        self.has_smoke_item = has_smoke_item
        
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.smoke_pos = (grid_size // 2, grid_size // 2) if has_smoke_item else None
        
        self.reset()
    
    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.hazard_pos = [self.grid_size - 1, 0]
        self.has_smoke = 1 if self.has_smoke_item else 0
        self.smoke_timer = 0
        return self._get_state()
    
    def _get_state(self):
        return (
            tuple(self.agent_pos),
            tuple(self.hazard_pos),
            self.has_smoke,
            self.smoke_timer
        )
        
    def _get_valid_moves(self, pos):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid = []
        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                valid.append((nr, nc))
        return valid
    
    def _move_hazard(self):
        valid_moves = self._get_valid_moves(self.hazard_pos)
        
        if self.smoke_timer > 0 or self.hazard_type == "random":
            next_move = random.choice(valid_moves)
        elif self.hazard_type == "chasing":
            dists = [abs(m[0] - self.agent_pos[0]) + abs(m[1] - self.agent_pos[1]) for m in valid_moves]
            min_dist = min(dists)
            
            weights = []
            for d in dists:
                if d == min_dist:
                    weights.append(0.6)
                else:
                    weights.append(0.4 / max(1, (len(valid_moves) -1)))
            
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            next_move = list(random.choices(valid_moves, weights = probs)[0])
        return next_move

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0 ,0)}
        dr, dc = moves[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc 
        
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            self.agent_pos = [nr, nc]
            
        if self.smoke_pos and tuple(self.agent_pos) == self.smoke_pos and self.has_smoke == 1:
                self.has_smoke = 0
                self.smoke_timer = 3
                print(">> AGENT ĐÃ ĂN SMOKE BOMB! Hazard bị hoảng loạn 3 bước. <<")
        elif self.smoke_timer > 0:
            self.smoke_timer -= 1
            
        self.hazard_pos = self._move_hazard()
        
        reward = -1
        done = False
        info = "Bình Thường"
        
        if self.agent_pos == self.hazard_pos:
            reward = -100
            done = True
            info = "THẤT BẠI: Va chạm Hazard!"
            
        elif tuple(self.agent_pos) == self.goal_pos:
            reward = 100
            done = True
            info = "THÀNH CÔNG: Đã tới Goal!"
        return self._get_state(), reward, done, info
    
def run_visual_demo():
    env = RandomHazardEnvDemo(grid_size=6, hazard_type="chasing", has_smoke_item=True)
    state = env.reset()
    done = False
    
    plt.ion() # Bật chế độ tương tác Matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))

    action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Wait"}
    step_count = 0

    while not done and step_count < 30:
        step_count += 1
        ax.clear()
        
        # Vẽ các thành phần trên sơ đồ Grid
        # 1. Goal (G)
        ax.text(env.goal_pos[1], env.goal_pos[0], 'GOAL', ha='center', va='center', color='green', weight='bold', fontsize=12)
        
        # 2. Smoke Item (S)
        if env.smoke_pos and env.has_smoke == 1:
            ax.text(env.smoke_pos[1], env.smoke_pos[0], 'SMOKE', ha='center', va='center', color='purple', weight='bold', fontsize=10)
            
        # 3. Agent (A)
        ax.text(env.agent_pos[1], env.agent_pos[0], 'A', ha='center', va='center', color='blue', weight='bold', fontsize=18)
        
        # 4. Hazard (H)
        hazard_color = 'orange' if env.smoke_timer > 0 else 'red'
        ax.text(env.hazard_pos[1], env.hazard_pos[0], 'H', ha='center', va='center', color=hazard_color, weight='bold', fontsize=18)

        # Cấu hình khung lưới
        ax.set_xticks(np.arange(-0.5, env.grid_size, 1))
        ax.set_yticks(np.arange(-0.5, env.grid_size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color='black', linewidth=1)
        ax.set_title(f"Step: {step_count} | Smoke Timer: {env.smoke_timer}", fontsize=12)
        plt.gca().invert_yaxis()
        plt.draw()
        plt.pause(0.6)

        # Cho Agent di chuyển ngẫu nhiên (hoặc thử nghiệm hành động bất kỳ)
        action = random.choice([0, 1, 2, 3, 4])
        state, reward, done, info = env.step(action)
        print(f"Step {step_count}: Action={action_names[action]} -> Reward={reward} | Status: {info}")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_visual_demo()