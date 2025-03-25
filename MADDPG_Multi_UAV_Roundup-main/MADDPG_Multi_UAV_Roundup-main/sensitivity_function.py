import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors


def generate_sensitivity_function(grid_size=(10, 10), random=True, smooth_factor=10.0):
    """
    生成一个二维敏感度函数，随机生成或使用预设的函数
    :param grid_size: 敏感度函数的网格大小
    :param random: 是否使用随机生成的敏感度
    :param smooth_factor: 高斯平滑的标准差，用于控制平滑程度，越大越平滑
    :return: 生成的敏感度矩阵
    """
    # 确保 smooth_factor 是有效的数值
    if smooth_factor is None or smooth_factor <= 0:
        raise ValueError("smooth_factor must be a positive number")

    if random:
        # 随机生成敏感度矩阵，值在[0, 1]之间
        sensitivity = np.random.rand(grid_size[0], grid_size[1])

        # 使用高斯滤波器平滑生成的矩阵，平滑程度由smooth_factor控制
        sensitivity = gaussian_filter(sensitivity, sigma=smooth_factor)
    else:
        # 预设敏感度函数，例如中心区域更重要，边缘区域重要性较低
        x = np.linspace(-1, 1, grid_size[0])
        y = np.linspace(-1, 1, grid_size[1])
        X, Y = np.meshgrid(x, y)
        sensitivity = np.exp(-(X ** 2 + Y ** 2))  # 高斯函数，中心区域值大，边缘值小

    # 确保敏感度值在[0, 1]之间，并将其转换为numpy数组的浮动类型
    sensitivity = np.clip(sensitivity, 0, 1)
    sensitivity = np.array(sensitivity, dtype=np.float64)  # 确保为float类型
    return sensitivity


def plot_sensitivity_map(sensitivity_matrix):
    """
    绘制敏感度矩阵的热力图
    :param sensitivity_matrix: 敏感度矩阵
    """
    plt.figure(figsize=(8, 6))
    # 创建颜色映射（红橙黄绿青蓝）
    #cmap = mcolors.ListedColormap(['#ff0000', '#ff7f00', '#ffff00', '#00ff00', '#00ffff', '#0000ff'])
    #bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 敏感度值的划分范围
    #norm = mcolors.BoundaryNorm(bounds, cmap.N)

    cmap = plt.get_cmap('viridis')  # 使用渐变色映射（如 viridis、plasma、coolwarm）
    norm = plt.Normalize(vmin=np.min(sensitivity_matrix), vmax=np.max(sensitivity_matrix))

    # 绘制热力图
    plt.imshow(sensitivity_matrix, cmap=cmap, norm=norm, origin='lower')
    plt.colorbar(label='Sensitivity Level')
    #plt.colorbar(boundaries=bounds, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.title('Environmental Sensitivity Map')
    plt.show()
if __name__ == '__main__':
    # 生成敏感度函数（可以是随机的或预设的）
    sensitivity_matrix = generate_sensitivity_function(grid_size=(100, 100), random=True)

    # 调用绘制敏感度函数的热力图
    plot_sensitivity_map(sensitivity_matrix)
