def dump_minerl_dataset(names, outfile):
    if not isinstance(names, list):
        names = [names]

    initial_size = 12500504

    dset_names = ['pov', 'vector', 'action', 'reward', 'done']

    f = h5py.File(outfile, 'w')
    f.create_dataset('pov', (initial_size, 64, 64, 3), maxshape=(None, 64, 64, 3), dtype='uint8')
    f.create_dataset('vector', (initial_size, 64), maxshape=(None, 64), dtype='f')
    f.create_dataset('action', (initial_size, 64), maxshape=(None, 64), dtype='f')
    f.create_dataset('reward', (initial_size, 1), maxshape=(None, 1), dtype='f', chunks=True)
    f.create_dataset('done', (initial_size, 1), maxshape=(None, 1), dtype='?', chunks=True)
    written = 0

    for name in names:
        minerl_dset = minerl.data.make(name, "data")

        for trajectory in tqdm(minerl_dset.get_trajectory_names()):
            pov = []
            vec = []
            act = []
            rew = []
            dones = []

            for i, (current_state, action, reward, next_state, done) in minerl_dset.load_data(trajectory):
                pov.append(current_state['pov'])
                vec.append(current_state['vector'])
                act.append(action['vector'])
                rew.append([reward])
                dones.append([done])

            size = len(pov)
            file_size = f['pov'].shape[0]
            if file_size - written < size:
                for dset_name in dset_names:
                    dset = f[dset_name]
                    dset.resize(file_size * 2, axis=0)

            f['pov'][written:written + size] = np.asarray(pov)
            f['vector'][written:written + size] = np.asarray(vec)
            f['action'][written:written + size] = np.asarray(act)
            f['reward'][written:written + size] = np.asarray(rew)
            f['done'][written:written + size] = np.asarray(dones)
            written += size

    # tidy up file sizes
    for dset_name in dset_names:
        dset = f[dset_name]
        dset.resize(written, axis=0)
    f.close()


def dump_minerl_dataset(names, outfile):
    if not isinstance(names, list):
        names = [names]

    initial_size = 12500504

    dset_names = ['pov', 'vector', 'action', 'reward', 'done']

    f = h5py.File(outfile, 'w')
    pov = f.create_dataset('pov', (initial_size, 64, 64, 3), maxshape=(initial_size, 64, 64, 3), dtype='uint8')
    vec = f.create_dataset('vector', (initial_size, 64), maxshape=(initial_size, 64), dtype='f')
    act = f.create_dataset('action', (initial_size, 64), maxshape=(initial_size, 64), dtype='f')
    rew = f.create_dataset('reward', (initial_size, 1), maxshape=(initial_size, 1), dtype='f')
    don = f.create_dataset('done', (initial_size, 1), maxshape=(initial_size, 1), dtype='?')
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

    f.close()


def dump_minerl_dataset(names):
    if not isinstance(names, list):
        names = [names]

    initial_size = 12500504

    pov = np.memmap('data/pov.npy', dtype='uint8', mode='w+', shape=(initial_size, 64, 64, 3))
    vec = np.memmap('data/vector.npy', dtype='f', mode='w+', shape=(initial_size, 64))
    act = np.memmap('data/action.npy', dtype='f', mode='w+', shape=(initial_size, 64))
    rew = np.memmap('data/reward.npy', dtype='f', mode='w+', shape=(initial_size, 1))
    don = np.memmap('data/done.npy', dtype='?', mode='w+', shape=(initial_size, 1))
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
