import os
import minerl
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

DATASET_SIZE = 1916597


def dump_minerl_dataset(names, output_dir):
    if not isinstance(names, list):
        names = [names]

    initial_size = 1916597
    
    os.makedirs(output_dir)

    pov = np.memmap(output_dir + '/pov.npy', dtype='uint8', mode='w+', shape=(initial_size, 64, 64, 3))
    vec = np.memmap(output_dir + '/vector.npy', dtype='f', mode='w+', shape=(initial_size, 64))
    act = np.memmap(output_dir + '/action.npy', dtype='f', mode='w+', shape=(initial_size, 64))
    rew = np.memmap(output_dir + '/reward.npy', dtype='f', mode='w+', shape=(initial_size, 1))
    don = np.memmap(output_dir + '/done.npy', dtype='?', mode='w+', shape=(initial_size, 1))
    written = 0

    for name in names:
        minerl_dset = minerl.data.make(name, "data")

        for trajectory in tqdm(minerl_dset.get_trajectory_names()):
            traj_data = list(minerl_dset.load_data(trajectory))

            for i, data in enumerate(traj_data):
                current_state, action, reward, next_state, done = data
                idx = written + i
                pov[idx] = current_state['pov']
                vec[idx] = current_state['vector']
                act[idx] = action['vector']
                rew[idx, 0] = reward
                don[idx, 0] = done

            size = len(traj_data)
            written += size
    return written


class MineRlSequenceDataset(Dataset):
    def __init__(self, base_dir, sequence_length):
        initial_size = 1916597
        self.pov = np.memmap(base_dir + '/pov.npy', dtype='uint8', mode='r', shape=(initial_size, 64, 64, 3))
        self.vec = np.memmap(base_dir + '/vector.npy', dtype='f', mode='r', shape=(initial_size, 64))
        self.act = np.memmap(base_dir + '/action.npy', dtype='f', mode='r', shape=(initial_size, 64))
        self.rew = np.memmap(base_dir + '/reward.npy', dtype='f', mode='r', shape=(initial_size, 1))
        self.don = np.memmap(base_dir + '/done.npy', dtype='?', mode='r', shape=(initial_size, 1))
        self.sequence_length = sequence_length

    def __len__(self):
        return self.pov.shape[0] - self.sequence_length + 1

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        pov = np.float32(self.pov[idx:end_idx].transpose(0, 3, 1, 2)) / 255
        vec = self.vec[idx:end_idx]
        act = self.act[idx:end_idx]
        rew = self.rew[idx:end_idx]
        don = self.don[idx:end_idx].astype('float32')
        return (pov, vec, act, rew, don)

    
class MineRlImageDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        initial_size = 1916597
        self.transform = transform
        self.pov = np.memmap(base_dir + '/pov.npy', dtype='uint8', mode='r', shape=(initial_size, 64, 64, 3))

    def __len__(self):
        return self.pov.shape[0]

    def __getitem__(self, idx):
        image = self.pov[idx]
        if self.transform:
            image = self.transform(image)
        return image