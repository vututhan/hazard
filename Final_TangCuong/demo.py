import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from env import RandomHazardEnv

# 1. Load Q-table đã huấn luyện
try:
    with open("trained_model.pkl", "rb") as f:
        data = pickle.load(f)
    q_table = data["q_table"]
except FileNotFoundError:
    print("Chưa tìm thấy file 'trained_model.pkl'. Hãy chạy train_and_plot.py trước!")
    q_table = {}

# 2. Khởi tạo môi trường
env = RandomHazardEnv(grid_size=6, hazard_type="chasing", num_hazards=1, has_smoke_item=True)
state, _ = env.reset()
done = False

# Bật chế độ tương tác Matplotlib
plt.ion()

# Tạo khung hình gồm 2 phần: Bên trái vẽ Grid Map, Bên phải hiện Lịch sử bước đi
fig, (ax_map, ax_log) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={'width_ratios': [1.2, 1]})

step_count = 0
action_names = {0: "Up ⬆️", 1: "Down ⬇️", 2: "Left ⬅️", 3: "Right ➡️", 4: "Wait ⏸️"}

# Danh sách lưu vết di chuyển
agent_history = [tuple(env.agent_pos)]
step_logs = []

while not done and step_count < 30:
    step_count += 1
    
    # --- PHẦN 1: XỬ LÝ HÀNH ĐỘNG DỰA TRÊN Q-TABLE ---
    if state in q_table:
        action = int(np.argmax(q_table[state]))
    else:
        action = np.random.randint(5)
        
    action_str = action_names[action]
    
    # Thực hiện bước đi
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    # Cập nhật thông tin vết di chuyển
    curr_pos = tuple(env.agent_pos)
    agent_history.append(curr_pos)
    step_logs.append(f"Step {step_count:02d}: {action_str:<8} -> {curr_pos}")

    # --- PHẦN 2: VẼ BẢN ĐỒ (GRID MAP) ---
    ax_map.clear()
    
    # 1. Vẽ Goal (G) & Smoke Item (S)
    ax_map.text(env.goal_pos[1], env.goal_pos[0], 'GOAL', ha='center', va='center', color='green', weight='bold', fontsize=11)
    if env.smoke_pos and env.has_smoke == 1:
        ax_map.text(env.smoke_pos[1], env.smoke_pos[0], 'SMOKE', ha='center', va='center', color='purple', weight='bold', fontsize=9)
        
    # 2. Vẽ đường vết di chuyển của Agent (Trajectory)
    if len(agent_history) > 1:
        hx = [p[1] for p in agent_history]
        hy = [p[0] for p in agent_history]
        ax_map.plot(hx, hy, color='dodgerblue', linestyle='--', linewidth=2, marker='o', markersize=4, alpha=0.7)

    # 3. Vẽ Agent (A)
    ax_map.text(env.agent_pos[1], env.agent_pos[0], 'A', ha='center', va='center', color='blue', weight='bold', fontsize=18)
    
    # 4. Vẽ Hazard (H)
    hazard_color = 'orange' if env.smoke_timer > 0 else 'red'
    for h in env.hazards_pos:
        ax_map.text(h[1], h[0], 'H', ha='center', va='center', color=hazard_color, weight='bold', fontsize=18)

    # Cấu hình khung lưới
    ax_map.set_xticks(np.arange(-0.5, env.grid_size, 1))
    ax_map.set_yticks(np.arange(-0.5, env.grid_size, 1))
    ax_map.set_xticklabels([])
    ax_map.set_yticklabels([])
    ax_map.grid(True, color='gray', linestyle=':', linewidth=1)
    ax_map.set_title(f"Grid Map | Step: {step_count} | Smoke: {env.smoke_timer}s", fontsize=12)
    ax_map.invert_yaxis()

    # --- PHẦN 3: HIỂN THỊ CÁC BƯỚC DI CHUYỂN BÊN PHẢI ---
    ax_log.clear()
    ax_log.axis('off')  # Ẩn trục tọa độ bảng log
    
    # Hiển thị tiêu đề bảng log
    log_text = "LỊCH SỬ BƯỚC DI CHUYỂN\n" + "="*30 + "\n"
    
    # Chỉ hiển thị 15 bước gần nhất để tránh tràn màn hình
    display_logs = step_logs[-15:]
    log_text += "\n".join(display_logs)
    
    if done:
        if reward > 0:
            log_text += "\n\n🎉 KẾT QUẢ: THÀNH CÔNG (Đạt Goal)!"
        else:
            log_text += "\n\n💥 KẾT QUẢ: THẤT BẠI (Chạm Hazard)!"

    ax_log.text(0.05, 0.95, log_text, transform=ax_log.transAxes,
                fontsize=10, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.8))

    plt.draw()
    plt.pause(0.5)

plt.ioff()
plt.show()