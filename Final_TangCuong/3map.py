import tkinter as tk
from tkinter import messagebox
import random

class GameEnvironment:
    def __init__(self, setting=1):
        self.setting = setting
        self.setup_setting_config()
        self.reset()

    def setup_setting_config(self):
        """Cấu hình 3 Map theo yêu cầu đề tài"""
        if self.setting == 1:
            # Setting 1: Map 5x5, 1 Hazard Random Walk thuần
            self.grid_size = 5
            self.start_pos = (0, 0)
            self.goal_pos = (4, 4)
            self.hazard_type = "random"
            self.num_hazards = 1
            self.has_smoke_item = False
            self.smoke_pos = None
            self.title = "Setting 1: Map 5x5 - 1 Hazard Random Walk"

        elif self.setting == 2:
            # Setting 2: Map 6x6, 2 Hazard độc lập
            self.grid_size = 6
            self.start_pos = (0, 0)
            self.goal_pos = (5, 5)
            self.hazard_type = "random"
            self.num_hazards = 2
            self.has_smoke_item = False
            self.smoke_pos = None
            self.title = "Setting 2: Map 6x6 - 2 Hazards Độc lập"

        elif self.setting == 3:
            # Setting 3: Map 6x6, 1 Hazard Chasing + Smoke Bomb
            self.grid_size = 6
            self.start_pos = (0, 0)
            self.goal_pos = (5, 5)
            self.hazard_type = "chasing"
            self.num_hazards = 1
            self.has_smoke_item = True
            self.smoke_pos = (3, 3)
            self.title = "Setting 3: Map 6x6 - 1 Chasing Hazard + Smoke Bomb"

    def reset(self):
        self.agent_pos = list(self.start_pos)
        
        # Khởi tạo vị trí Hazard
        if self.num_hazards == 1:
            self.hazards_pos = [[self.grid_size - 1, 0]]
        else:
            self.hazards_pos = [[self.grid_size - 1, 0], [0, self.grid_size - 1]]

        self.has_smoke = 1 if self.has_smoke_item else 0
        self.smoke_timer = 0
        self.step_count = 0
        self.game_over = False

    def get_valid_moves(self, pos):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid = []
        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                valid.append((nr, nc))
        return valid

    def move_hazard(self, h_pos):
        valid_moves = self.get_valid_moves(h_pos)
        
        # Nếu Smoke bomb đang hoạt động hoặc hazard_type="random" -> Random Walk
        if self.smoke_timer > 0 or self.hazard_type == "random":
            return list(random.choice(valid_moves))
        
        # Hazard đuổi theo Agent (Chasing Mode)
        elif self.hazard_type == "chasing":
            dists = [abs(m[0] - self.agent_pos[0]) + abs(m[1] - self.agent_pos[1]) for m in valid_moves]
            min_dist = min(dists)
            
            weights = []
            for d in dists:
                if d == min_dist:
                    weights.append(0.6)  # 60% xác suất đuổi gần
                else:
                    weights.append(0.4 / max(1, (len(valid_moves) - 1)))
            
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            return list(random.choices(valid_moves, weights=probs)[0])

    def step(self, action_move):
        """Action: (-1,0): Up, (1,0): Down, (0,-1): Left, (0,1): Right, (0,0): Wait"""
        if self.game_over:
            return

        self.step_count += 1
        dr, dc = action_move
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # 1. Di chuyển Agent
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            self.agent_pos = [nr, nc]

        # 2. Xử lý vật phẩm Smoke Bomb
        if self.smoke_pos and tuple(self.agent_pos) == self.smoke_pos and self.has_smoke == 1:
            self.has_smoke = 0
            self.smoke_timer = 3  # Tác dụng 3 bước
        elif self.smoke_timer > 0:
            self.smoke_timer -= 1

        # 3. Di chuyển Hazard
        for i in range(len(self.hazards_pos)):
            self.hazards_pos[i] = self.move_hazard(self.hazards_pos[i])

        # 4. Kiểm tra điều kiện thắng / thua
        for h in self.hazards_pos:
            if self.agent_pos == h:
                self.game_over = True
                return "LOSE"

        if tuple(self.agent_pos) == self.goal_pos:
            self.game_over = True
            return "WIN"

        return "PLAYING"


class GameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RL Game - Đề tài 23: Random Moving Hazard")
        self.current_setting = 1
        self.env = GameEnvironment(setting=self.current_setting)

        # Thanh chọn Map / Setting
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10)

        tk.Label(top_frame, text="Chọn Bản Đồ: ", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        tk.Button(top_frame, text="Setting 1 (5x5)", command=lambda: self.change_map(1), bg="#E1F5FE").pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Setting 2 (6x6)", command=lambda: self.change_map(2), bg="#FFF9C4").pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Setting 3 (6x6 + Smoke)", command=lambda: self.change_map(3), bg="#F8BBD0").pack(side=tk.LEFT, padx=5)

        # Tiêu đề Map
        self.lbl_title = tk.Label(self.root, text="", font=("Arial", 11, "bold"), fg="blue")
        self.lbl_title.pack()

        # Khung chứa Canvas vẽ ô lưới
        self.canvas = tk.Canvas(self.root, width=400, height=400, bg="white")
        self.canvas.pack(pady=10)

        # Thông tin trạng thái
        self.lbl_status = tk.Label(self.root, text="", font=("Arial", 10))
        self.lbl_status.pack()

        # Hướng dẫn điều khiển
        guide_text = "Điều khiển: ⬆️ ⬇️ ⬅️ ➡️ để di chuyển | Spacebar: Đứng chờ (Wait) | R: Chơi lại"
        tk.Label(self.root, text=guide_text, font=("Arial", 9, "italic"), fg="gray").pack(pady=5)

        # Gắn phím điều khiển
        self.root.bind("<Up>", lambda event: self.play_step((-1, 0)))
        self.root.bind("<Down>", lambda event: self.play_step((1, 0)))
        self.root.bind("<Left>", lambda event: self.play_step((0, -1)))
        self.root.bind("<Right>", lambda event: self.play_step((0, 1)))
        self.root.bind("<space>", lambda event: self.play_step((0, 0)))
        self.root.bind("<r>", lambda event: self.reset_game())

        self.update_gui()

    def change_map(self, setting):
        self.current_setting = setting
        self.env = GameEnvironment(setting=setting)
        self.update_gui()

    def reset_game(self):
        self.env.reset()
        self.update_gui()

    def play_step(self, action_move):
        if self.env.game_over:
            return
        
        status = self.env.step(action_move)
        self.update_gui()

        if status == "WIN":
            messagebox.showinfo("Kết quả", "🎉 THÀNH CÔNG! Bạn đã đến được Goal an toàn!")
        elif status == "LOSE":
            messagebox.showerror("Kết quả", "💥 THẤT BẠI! Bạn đã bị Hazard đâm phải!")

    def update_gui(self):
        self.canvas.delete("all")
        self.lbl_title.config(text=self.env.title)

        grid_size = self.env.grid_size
        cell_size = 400 // grid_size

        # Vẽ lưới
        for r in range(grid_size):
            for c in range(grid_size):
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray", fill="#FAFAFA")

        # 1. Vẽ Goal (Xanh lá)
        gr, gc = self.env.goal_pos
        self.canvas.create_rectangle(gc * cell_size, gr * cell_size, (gc + 1) * cell_size, (gr + 1) * cell_size, fill="#C8E6C9")
        self.canvas.create_text((gc + 0.5) * cell_size, (gr + 0.5) * cell_size, text="GOAL", font=("Arial", 10, "bold"), fill="green")

        # 2. Vẽ Smoke Item (Tím)
        if self.env.smoke_pos and self.env.has_smoke == 1:
            sr, sc = self.env.smoke_pos
            self.canvas.create_oval(sc * cell_size + 5, sr * cell_size + 5, (sc + 1) * cell_size - 5, (sr + 1) * cell_size - 5, fill="#E1BEE7")
            self.canvas.create_text((sc + 0.5) * cell_size, (sr + 0.5) * cell_size, text="SMOKE", font=("Arial", 8, "bold"), fill="purple")

        # 3. Vẽ Agent (Xanh dương)
        ar, ac = self.env.agent_pos
        self.canvas.create_oval(ac * cell_size + 6, ar * cell_size + 6, (ac + 1) * cell_size - 6, (ar + 1) * cell_size - 6, fill="#2196F3")
        self.canvas.create_text((ac + 0.5) * cell_size, (ar + 0.5) * cell_size, text="A", font=("Arial", 12, "bold"), fill="white")

        # 4. Vẽ Hazard (Đỏ / Cam khi bị dính Bom Khói)
        hazard_color = "#FF9800" if self.env.smoke_timer > 0 else "#F44336"
        for hr, hc in self.env.hazards_pos:
            self.canvas.create_rectangle(hc * cell_size + 8, hr * cell_size + 8, (hc + 1) * cell_size - 8, (hr + 1) * cell_size - 8, fill=hazard_color)
            self.canvas.create_text((hc + 0.5) * cell_size, (hr + 0.5) * cell_size, text="H", font=("Arial", 12, "bold"), fill="white")

        # Cập nhật Label trạng thái
        status_text = f"Bước: {self.env.step_count} | Bom Khói còn lại: {'Có' if self.env.has_smoke else 'Đã dùng'} | Thời gian khói: {self.env.smoke_timer}s"
        self.lbl_status.config(text=status_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = GameGUI(root)
    root.mainloop()