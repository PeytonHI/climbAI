# dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PoseSequenceDataset(Dataset):
    """
    Loads multiple videos' pose .npy files and returns sequences of length seq_len.
    Each pose frame is shape (33,4) -> we use only (x,y) and visibility optionally.
    """

    def __init__(self, root_dir, seq_len=64, use_visibility=True, stride=8):
        """
        root_dir: folder containing subfolders each with poses.npy
        seq_len: number of frames per sequence
        stride: sliding window stride (controls how many sequences per file)
        """
        self.seq_len = seq_len
        self.use_visibility = use_visibility
        self.stride = stride

        self.files = []
        for sub in os.listdir(root_dir):
            p = os.path.join(root_dir, sub, "poses.npy")
            if os.path.exists(p):
                self.files.append(p)
        self.indices = []  # list of tuples (file_idx, start_frame)

        for i,f in enumerate(self.files):
            arr = np.load(f, mmap_mode='r')
            n = arr.shape[0]
            for s in range(0, n - seq_len + 1, stride):
                self.indices.append((i, s))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, start = self.indices[idx]
        arr = np.load(self.files[file_idx])
        seq = arr[start:start+self.seq_len]  # (T, 33, 4)
        # take x,y normalized by image coordinates since mediapipe gives normalized [0,1]
        xy = seq[..., :2]  # (T, 33, 2)
        if self.use_visibility:
            vis = seq[..., 3:4]  # (T,33,1)
            data = np.concatenate([xy, vis], axis=-1)  # (T,33,3)
        else:
            data = xy
        # flatten landmarks: (T, 33*D)
        T, L, D = data.shape
        data = data.reshape(T, L*D)
        # optionally normalize per sequence (center hip)
        # center on mid-hip (landmarks 23,24 in mediapipe, or check numbering)
        mid_hip = (seq[:,23,:2] + seq[:,24,:2]) / 2.0  # (T,2)
        data_xy = data.reshape(T, L, D)
        for t in range(T):
            data_xy[t,:,:2] = data_xy[t,:,:2] - mid_hip[t:t+1]  # center
        data = data_xy.reshape(T, L*D)
        return torch.from_numpy(data).float()
