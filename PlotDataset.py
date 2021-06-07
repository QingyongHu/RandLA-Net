import numpy
import numpy as np
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

Path = r'/data/semantic_kitti/dataset/sequences_0.12'


def load_points(seq_id):
    points = []
    kd_tree_path = os.path.join(Path, seq_id, 'KDTree')
    for d, _, f in os.walk(kd_tree_path):
        for nf in tqdm(f):
            with open(os.path.join(d, nf), 'rb') as of:
                search_tree = pickle.load(of)
            points.append(np.array(search_tree.data, copy=True))
    return np.array(points, dtype=object)


def get_dists(pts):
    ret = []
    for p in pts:
        ret.append(numpy.dot(p, p))
    return np.array(ret)


def plot_xyz(pts):
    ax = plt.subplot(projection='3d')
    ax.set_title('Points')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


a = []

if __name__ == "__main__":
    a = load_points("00")
    c=np.zeros((10000,))
    for i in range(a.shape[0]):

    c = np.bincount(d)
    x = np.arange(0, c.shape[0])
    plt.plot(x, c)
    plt.show()
