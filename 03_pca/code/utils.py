import yaml
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib

matplotlib.use("TkAgg")


# load .mat file to numpy array
def LoadDataFromMat(file_name: str) -> np.array:
    return sio.loadmat(file_name)['data']


# loads data from yaml file
def LoadDataFromYaml(file_name: str) -> dict:
    with open(file_name, 'r') as stream:
        return yaml.safe_load(stream)


# loads data from txt file to numpy array
def LoadDataFromTxt(file_name: str) -> np.array:
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            data.append(line.split())
    return np.array(data, dtype=float)


# split numpy array into 3 parts (X, Y, Z) by columns and return as dictionary
def SplitData(data: np.array) -> dict:
    X = data[:, ::3]
    Y = data[:, 1::3]
    Z = data[:, 2::3]
    return {'X': X, 'Y': Y, 'Z': Z}


# find axis limits for 3d plot
def FindLimits(data: dict) -> dict:
    x_min = np.min(data['X'])
    x_max = np.max(data['X'])
    y_min = np.min(data['Y'])
    y_max = np.max(data['Y'])
    z_min = np.min(data['Z'])
    z_max = np.max(data['Z'])
    return {'X': (x_min, x_max), 'Y': (y_min, y_max), 'Z': (z_min, z_max)}


# plot 3d animation with matplotlib from numpy array
def Plot3DAnimation(data: dict, connections: dict, limits: dict) -> None:
    fig = plt.figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot(data['X'][0, :], data['Y'][0, :], data['Z'][0, :], 'o')
    ax.view_init(azim=0, elev=0)

    def animate(i: int):
        plt.cla()
        ax.set_xlim3d(limits['X'][0], limits['X'][1])
        ax.set_ylim3d(limits['Y'][0], limits['Y'][1])
        ax.set_zlim3d(limits['Z'][0], limits['Z'][1])
        ax.plot(data['X'][i, connections], data['Y'][i, connections], data['Z'][i, connections], '-')

    anim = animation.FuncAnimation(fig, animate, frames=data['X'][:, 0].size, interval=0.001)
    plt.show()


# concatenate each numpy array in dict with NaN column
def AppendNaN(data: dict) -> dict:
    for key in data:
        data[key] = np.concatenate((data[key], np.full((data[key].shape[0], 1), np.nan)), axis=1)
    return data


# load text file with ignoring lines with % and deleting new lines characters and return list of ints with subtracted 1
def LoadDataFromTxtIgnoreComments(file_name: str) -> list:
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            if line[0] != '%':
                data.append(int(line.replace('\n', '')))
    return SubtractFromList(data, 1)


# subtract from list constant value
def SubtractFromList(list: list, value: int) -> list:
    return [x - value for x in list]


def main():
    data = LoadDataFromTxt('../data/walk1.txt')
    data_dict = AppendNaN(SplitData(data))
    connections = LoadDataFromTxtIgnoreComments('../data/connected_points.txt')
    limits = FindLimits(SplitData(data))
    Plot3DAnimation(data_dict, connections, limits)


if __name__ == "__main__":
    main()
    pass
