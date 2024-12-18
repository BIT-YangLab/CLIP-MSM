import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm

def calculate_density(data, x_divided = 100, y_divided = 100):
    # X_min, X_max = data[:,0].min(), data[:,0].max()
    # Y_min, Y_max = data[:,1].min(), data[:,1].max()

    X_min, X_max, Y_min, Y_max = 0, 1, 0, 1
    x_window = (X_max - X_min) / x_divided
    y_window = (Y_max - Y_min) / y_divided
    # print(X_min, X_max)
    x_position = np.arange(X_min, X_max, x_window)
    y_position = np.arange(Y_min, Y_max, y_window)

    # x_position = [1,2,3]
    # y_position = [3,4,5]

    x,y = np.array(np.meshgrid(x_position, y_position, indexing='ij'))
    position = np.column_stack((x.ravel(), y.ravel()))

    print("pos:",position.shape, len(position))
    print(position)
    density = np.zeros(len(position))
    # density = np.zeros_like(data)
    for i in range(len(position)):
        x_min, x_max = position[i][0] - x_window/2, position[i][0] + x_window/2
        y_min, y_max = position[i][1] - y_window/2, position[i][1] + y_window/2
        mask = (data[:, 0] >= x_min) & (data[:, 0] <= x_max) & \
               (data[:, 1] >= y_min) & (data[:, 1] <= y_max)
        density[i] = np.sum(mask)
    return position, density

def scatter(data, x_divided = 100, y_divided = 100, s = 5, save = ""):
    plt.cla()
    
    position, density = calculate_density(data, x_divided, y_divided)
    norm_density = density / np.max(density)
    cmap = plt.cm.viridis

    mask = norm_density==0
    position = position[~mask]
    norm_density = norm_density[~mask]

    plt.scatter(position[:, 0], position[:, 1], c=norm_density, cmap=cmap, alpha=1,s=s)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=LogNorm(vmin = 1, vmax = np.max(density)))
    cb = plt.colorbar(sm)

    cb.ax.tick_params(labelsize=12)
    cb.ax.tick_params(labelright=False)
    print("save_path:", save)

    plt.xlim(0,1)
    plt.ylim(0,1)

    x = np.linspace(0,1,50)
    # print(x)
    y1, y2 = x, 0.85*x

    plt.plot(x, y1, label='Noise ceiling', color='red', linewidth = 4)
    plt.plot(x, y2, label='85% noise ceiling', linestyle = '--', color='orange', linewidth = 4)

    # plt.legend(fontsize = 14)

    # plt.xlabel('Noise ceiling', fontsize = 15)
    # plt.ylabel('Model performance (R²)', fontsize = 15)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tick_params(labelbottom=False, labelleft=False)

    plt.tight_layout()

    ax = plt.gca()

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    if save != "":
        print("save_path:", save)
        plt.savefig(save)
    plt.show()