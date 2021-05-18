import numpy as np
from tqdm import tqdm
import os

MAXL = 17
s = np.zeros(MAXL, np.int64)
for seqid in range(11):
    dirpath = '/data/semantic_kitti/dataset/sequences_0.12/{:02d}/labels'.format(seqid)
    # dirpath = '/data/kitti/dataset/sequences_0.06/{:02d}/labels'.format(seqid)
    if not os.path.isdir(dirpath):
        continue
    print('sequence {}'.format(seqid))
    for f in tqdm(os.listdir(dirpath)):
        lb = np.load(os.path.join(dirpath, f)).squeeze()
        # lb = np.fromfile(os.path.join(dirpath, f), dtype=np.uint32).reshape(-1) & 0xFFFF
        s += np.bincount(lb, minlength=MAXL)
print(dirpath)
print(s.__repr__())
