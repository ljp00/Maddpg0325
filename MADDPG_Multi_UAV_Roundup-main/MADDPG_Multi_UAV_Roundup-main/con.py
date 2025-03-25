import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors


def generate_sensitivity_function(grid_size=(10, 10), random=True, smooth_factor=15.0):  # 修改默认平滑系数
    if smooth_factor is None or smooth_factor <= 0:
        raise ValueError("smooth_factor must be a positive number")

    if random:
        sensitivity = np.random.rand(grid_size[0], grid_size[1])
        sensitivity = gaussian_filter(sensitivity, sigma=smooth_factor)

        # 新增归一化操作
        sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min())
    else:
        x = np.linspace(-1, 1, grid_size[0])
        y = np.linspace(-1, 1, grid_size[1])
        X, Y = np.meshgrid(x, y)
        sensitivity = np.exp(-(X ** 2 + Y ** 2))

    sensitivity = np.clip(sensitivity, 0, 1)
    sensitivity = np.array(sensitivity, dtype=np.float64)
    return sensitivity


def plot_sensitivity_map(sensitivity_matrix):
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=1)  # 固定范围为0-1以显示完整颜色区间
    plt.imshow(sensitivity_matrix, cmap=cmap, norm=norm, origin='lower')
    plt.colorbar(label='Sensitivity Level')
    plt.title('Environmental Sensitivity Map')
    plt.show()


if __name__ == '__main__':
    # 示例调用（使用更小的smooth_factor）
    sensitivity_matrix = generate_sensitivity_function(grid_size=(100, 100), random=True, smooth_factor=15.0)
    plot_sensitivity_map(sensitivity_matrix)