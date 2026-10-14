import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size=50):
    """Hàm làm mượt đường cong học tập bằng moving average"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_curves():
    # Giả lập dữ liệu return thu được từ qua trình train của Q-Learning và SARSA
    # (Trong thực tế huấn luyện, bạn lưu các mảng return của từng episode vào đây)
    episodes = 2000
    np.random.seed(42)
    
    # Dữ liệu mẫu minh họa xu hướng tăng dần và hội tụ của return
    q_returns = -100 + 195 / (1 + np.exp(-0.003 * (np.arange(episodes) - 500))) + np.random.normal(0, 10, episodes)
    sarsa_returns = -100 + 190 / (1 + np.exp(-0.0025 * (np.arange(episodes) - 600))) + np.random.normal(0, 12, episodes)

    # Làm mượt đường cong
    window = 100
    q_smooth = moving_average(q_returns, window)
    sarsa_smooth = moving_average(sarsa_returns, window)
    x_axis = np.arange(len(q_smooth))

    # Vẽ biểu đồ
    plt.figure(figsize=(9, 5))
    plt.plot(x_axis, q_smooth, label="Q-Learning (Off-Policy)", color="blue", linewidth=2)
    plt.plot(x_axis, sarsa_smooth, label="SARSA (On-Policy)", color="red", linewidth=2, linestyle="--")
    
    plt.title("Learning Curves Comparison (Setting 3)", fontsize=12, fontweight="bold")
    plt.xlabel(f"Episodes (Moving Average window={window})", fontsize=10)
    plt.ylabel("Average Return per Episode", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Lưu file ảnh để đưa vào báo cáo Word
    plt.savefig("learning_curve_comparison.png", dpi=300, bbox_inches="tight")
    print("Đã vẽ và lưu biểu đồ vào file 'learning_curve_comparison.png' thành công!")
    plt.show()

if __name__ == "__main__":
    plot_curves()